"""
============================================================================
CONFIG - Project Configuration
============================================================================

This file defines project paths and default hyperparameters.

Author: Ashutosh Kumar
============================================================================
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root (parent of rl_capstone folder)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Data directories (not used in V1, but kept for cookiecutter compatibility)
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Models directory - where we save trained agents
MODELS_DIR = PROJ_ROOT / "models"
POLICY_KERNEL_DIR = MODELS_DIR / "policy_kernel"

# Reports directory - for figures and results
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# =============================================================================
# DEFAULT HYPERPARAMETERS
# =============================================================================

DEFAULT_GAMMA = 0.99           # Discount factor
DEFAULT_B = 0.5                # Wind probability for Windy Chasm
DEFAULT_TOL = 1e-10            # Convergence tolerance
DEFAULT_MAX_ITER = 50000       # Maximum iterations


# Print project root on import (helpful for debugging)
print(f"Project root: {PROJ_ROOT}")
