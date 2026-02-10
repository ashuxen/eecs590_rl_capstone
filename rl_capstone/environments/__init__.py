"""
EECS 590 RL Capstone - Environments Module
==========================================

Custom RL environments for the capstone project.

Environments:
- WindyChasmEnv: Mini 2 grid-world with stochastic wind
- UR5eCableInsertionEnv: AI for Industry Challenge (preliminary)
"""

from .windy_chasm import WindyChasmEnv, build_windy_chasm_mdp
from .ur5e_cable_insertion import (
    UR5eCableInsertionEnv,
    UR5eConfig,
    RobotiqHandEConfig,
    Axia80FTConfig,
    CableInsertionTask
)

__all__ = [
    # Mini 2
    "WindyChasmEnv", 
    "build_windy_chasm_mdp",
    # AI for Industry Challenge
    "UR5eCableInsertionEnv",
    "UR5eConfig",
    "RobotiqHandEConfig",
    "Axia80FTConfig",
    "CableInsertionTask",
]
