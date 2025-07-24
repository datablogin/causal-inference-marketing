"""Causal Inference Library for Marketing Applications.

A comprehensive library for causal inference methods applied to marketing analytics,
following monorepo-compatible patterns for future integration.
"""

__version__ = "0.1.0"
__author__ = "Robert Welborn"

from .core import *
from .data import *
from .utils import *

__all__ = [
    "__version__",
    "__author__",
]
