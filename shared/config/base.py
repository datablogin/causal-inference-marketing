"""Base configuration management extracted from analytics-backend-monorepo."""

from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar("T", bound="BaseConfiguration")


class Environment(str, Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationAuditLog(BaseModel):
    """Audit log entry for configuration changes."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    environment: Environment
    key: str
    old_value: Any = None
    new_value: Any
    source: str
    user: str | None = None
    reason: str | None = None


class BaseConfiguration(BaseSettings):
    """Base configuration class compatible with monorepo patterns."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core metadata
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current deployment environment",
    )
    version: str = Field(default="1.0.0", description="Configuration version")
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last configuration update timestamp",
    )

    # Audit and monitoring
    audit_enabled: bool = Field(
        default=True, description="Enable configuration audit logging"
    )

    def to_dict(self, exclude_sensitive: bool = True) -> dict[str, Any]:
        """Convert configuration to dictionary, optionally excluding sensitive data."""
        data = self.model_dump()

        if exclude_sensitive:
            # Remove sensitive fields
            sensitive_patterns = ["password", "token", "key", "secret"]
            filtered_data = {}
            for k, v in data.items():
                if not any(pattern in k.lower() for pattern in sensitive_patterns):
                    filtered_data[k] = v
                else:
                    filtered_data[k] = "***REDACTED***"
            return filtered_data

        return data

    def validate_configuration(self) -> list[str]:
        """Validate the current configuration and return any issues."""
        issues = []

        # Basic validation - override in subclasses for specific validation
        if self.environment == Environment.PRODUCTION:
            # Add production-specific validations
            pass

        return issues


class ConfigurationManager:
    """Central configuration manager for all application configurations."""

    def __init__(self):
        self._configurations: dict[str, BaseConfiguration] = {}

    def register_configuration(self, name: str, config: BaseConfiguration) -> None:
        """Register a configuration instance."""
        self._configurations[name] = config

    def get_configuration(self, name: str) -> BaseConfiguration | None:
        """Get a registered configuration by name."""
        return self._configurations.get(name)

    def get_all_configurations(self) -> dict[str, dict[str, Any]]:
        """Get all configurations as dictionaries."""
        return {
            name: config.to_dict(exclude_sensitive=True)
            for name, config in self._configurations.items()
        }

    def validate_all_configurations(self) -> dict[str, list[str]]:
        """Validate all registered configurations."""
        validation_results = {}
        for name, config in self._configurations.items():
            issues = config.validate_configuration()
            if issues:
                validation_results[name] = issues
        return validation_results


# Global configuration manager instance
config_manager = ConfigurationManager()
