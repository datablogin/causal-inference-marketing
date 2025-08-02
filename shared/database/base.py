"""Database management compatible with analytics-backend-monorepo patterns."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from shared.config import CausalInferenceConfig


class DatabaseManager:
    """Database manager compatible with monorepo patterns."""

    def __init__(self, config: CausalInferenceConfig):
        self.config = config
        self._engine: Engine | None = None
        self._async_engine: AsyncEngine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._async_session_factory: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> Engine:
        """Get synchronous database engine."""
        if self._engine is None:
            # Convert async URL to sync URL if needed
            url = self.config.database_url
            if url.startswith("sqlite+aiosqlite://"):
                url = url.replace("sqlite+aiosqlite://", "sqlite://")
            elif url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql+asyncpg://", "postgresql://")

            self._engine = create_engine(
                url,
                pool_size=self.config.database_pool_size,
                echo=self.config.environment.value == "development",
            )
        return self._engine

    @property
    def async_engine(self) -> AsyncEngine:
        """Get asynchronous database engine."""
        if self._async_engine is None:
            # Ensure async URL format
            url = self.config.database_url
            if url.startswith("sqlite://"):
                url = url.replace("sqlite://", "sqlite+aiosqlite://")
            elif url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://")

            self._async_engine = create_async_engine(
                url,
                pool_size=self.config.database_pool_size,
                echo=self.config.environment.value == "development",
            )
        return self._async_engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Get synchronous session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    @property
    def async_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get asynchronous session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
            )
        return self._async_session_factory

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session context manager."""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_session(self) -> Session:
        """Get synchronous database session context manager."""
        return self.session_factory()

    async def close(self) -> None:
        """Close database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._engine:
            self._engine.dispose()


# Global database manager instance
_database_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database_manager
    if _database_manager is None:
        from shared.config import CausalInferenceConfig

        config = CausalInferenceConfig()
        _database_manager = DatabaseManager(config)
    return _database_manager
