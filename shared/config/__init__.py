"""Configuration management compatible with analytics-backend-monorepo."""

from .base import (
    BaseConfiguration,
    ConfigurationManager,
    Environment,
    config_manager,
)
from .causal_config import CausalInferenceConfig

__all__ = [
    "BaseConfiguration",
    "ConfigurationManager",
    "Environment",
    "config_manager",
    "CausalInferenceConfig",
]
