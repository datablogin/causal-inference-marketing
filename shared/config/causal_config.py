"""Causal inference specific configuration."""

from pydantic import Field, validator

from .base import BaseConfiguration, Environment


class CausalInferenceConfig(BaseConfiguration):
    """Configuration for causal inference services."""

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///causal_inference.db", description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=5, description="Database connection pool size"
    )

    # Computation Configuration
    max_sample_size: int = Field(
        default=1_000_000,
        description="Maximum sample size for causal inference computations",
    )
    computation_timeout: int = Field(
        default=300, description="Computation timeout in seconds"
    )
    enable_parallel_processing: bool = Field(
        default=True, description="Enable parallel processing for computations"
    )

    # Model Configuration
    default_confidence_level: float = Field(
        default=0.95, description="Default confidence level for causal estimates"
    )
    bootstrap_samples: int = Field(
        default=1000,
        description="Number of bootstrap samples for uncertainty estimation",
    )

    # Cache Configuration
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics endpoint port")

    @validator("default_confidence_level")
    def validate_confidence_level(cls, v: float) -> float:  # noqa: N805
        if not 0 < v < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        return v

    @validator("bootstrap_samples")
    def validate_bootstrap_samples(cls, v: int) -> int:  # noqa: N805
        if v < 100:
            raise ValueError("Bootstrap samples must be at least 100")
        return v

    def validate_configuration(self) -> list[str]:
        """Validate causal inference specific configuration."""
        issues = super().validate_configuration()

        # Production-specific validations
        if self.environment == Environment.PRODUCTION:
            if self.database_url.startswith("sqlite://"):
                issues.append("SQLite database not recommended for production")
            if self.api_workers < 2:
                issues.append("Consider using multiple API workers in production")
            if not self.enable_caching:
                issues.append("Caching should be enabled in production for performance")

        # Resource validations
        if self.max_sample_size > 10_000_000:
            issues.append("Very large sample sizes may cause memory issues")

        if self.computation_timeout < 60:
            issues.append("Computation timeout might be too short for complex analyses")

        return issues
