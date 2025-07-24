"""Health check endpoints."""

from typing import Dict, Any
from fastapi import APIRouter

from shared.config import config_manager
from shared.database import get_database_manager

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {"status": "healthy", "service": "causal-inference-api"}


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check including dependencies."""
    config = config_manager.get_configuration("causal_inference")
    
    # Check database connectivity
    db_status = "healthy"
    try:
        db_manager = get_database_manager()
        # Simple connection test
        with db_manager.get_session() as session:
            session.execute("SELECT 1")
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ready" if db_status == "healthy" else "not ready",
        "environment": config.environment.value if config else "unknown",
        "database": db_status,
        "version": "0.1.0"
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check."""
    return {"status": "alive", "timestamp": "2024-01-01T00:00:00Z"}