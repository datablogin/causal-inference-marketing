"""Target Trial Emulation Framework for causal inference.

This module provides a comprehensive framework for emulating randomized controlled trials
using observational data. It implements the target trial emulation approach described
in Chapter 22 of "Causal Inference: What If" by Hernán and Robins.

The framework helps ensure clear causal questions and appropriate analytical methods
by explicitly specifying the hypothetical randomized trial that would answer the
causal question of interest, then using observational data to emulate that trial.
"""

from .emulator import TargetTrialEmulator
from .protocol import TargetTrialProtocol
from .results import TargetTrialResults

__all__ = [
    "TargetTrialProtocol",
    "TargetTrialEmulator",
    "TargetTrialResults",
]
