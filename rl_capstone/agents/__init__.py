"""
============================================================================
AGENTS MODULE - RL Agents and Training Framework
============================================================================

This module contains the agent classes and training infrastructure.

FILES IN THIS MODULE:
---------------------

1. base_agent.py - BaseAgent (Abstract Class)
   - Defines what ALL agents must be able to do
   - select_action(), train(), save(), load()
   - You don't use this directly, but all agents inherit from it

2. dp_agent.py - DPAgent (Dynamic Programming Agent)
   - Agent that solves MDPs using DP algorithms
   - Supports: Value Iteration, Q-Value Iteration, Policy Iteration
   - This is what you use for Mini 2!

3. trainer.py - Trainer
   - Manages training and evaluation
   - Saves/loads models
   - Tracks history and best results

TYPICAL WORKFLOW:
-----------------
    from rl_capstone.agents import DPAgent, Trainer
    from rl_capstone.environments import WindyChasmEnv
    
    # 1. Create environment and agent
    env = WindyChasmEnv(B=0.5)
    agent = DPAgent(env, method="value_iteration")
    
    # 2. Create trainer
    trainer = Trainer(agent, env)
    
    # 3. Train (this solves the MDP)
    trainer.train()
    
    # 4. Evaluate (run episodes with learned policy)
    results = trainer.evaluate(num_episodes=100)
    print(f"Success rate: {results.success_rate:.1%}")
    
    # 5. Save for later
    trainer.save("models/policy_kernel/my_agent.pkl")

============================================================================
"""

from .base_agent import BaseAgent
from .dp_agent import DPAgent
from .trainer import Trainer

__all__ = ["BaseAgent", "DPAgent", "Trainer"]
