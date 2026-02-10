"""
============================================================================
EECS 590 RL CAPSTONE - Main Package
============================================================================

Welcome to the EECS 590 Reinforcement Learning Capstone project!

This package implements all V1 requirements:
  1. Cookiecutter project structure (you're looking at it!)
  2. Documentation with citations (see README.md)
  3. MDP representation with beliefs (see mdp/ module)
  4. Dynamic Programming algorithms (see algorithms/ module)
  5. Agent framework (see agents/ module)

PROJECT STRUCTURE:
------------------

    rl_capstone/
    │
    ├── mdp/              ← MDP classes (the math)
    │   ├── finite_mrp.py    - Markov Reward Process
    │   ├── finite_mdp.py    - Markov Decision Process  
    │   └── belief_mdp.py    - Model-based beliefs
    │
    ├── algorithms/       ← DP algorithms (the solving)
    │   ├── value_iteration.py
    │   ├── q_value_iteration.py
    │   ├── policy_iteration.py
    │   └── td_lambda.py
    │
    ├── agents/           ← Agent classes (the learning)
    │   ├── base_agent.py
    │   ├── dp_agent.py
    │   └── trainer.py
    │
    ├── environments/     ← Custom environments
    │   └── windy_chasm.py   - Mini 2 Problem 1
    │
    └── visualization/    ← Isaac Sim visualization
        └── windy_chasm_interactive.py

QUICK START:
------------

    # Import what is needed
    from rl_capstone.environments import WindyChasmEnv
    from rl_capstone.agents import DPAgent, Trainer
    
    # Create and train
    env = WindyChasmEnv(B=0.5, gamma=0.99)
    agent = DPAgent(env, method="value_iteration")
    trainer = Trainer(agent, env)
    
    # Train (solves the MDP)
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate(num_episodes=100)
    print(f"Success rate: {results.success_rate:.1%}")
    
    # Save
    trainer.save("models/policy_kernel/v1.pkl")

============================================================================
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Ashutosh Kumar"
__course__ = "EECS 590 Reinforcement Learning"

# Import config (from cookiecutter template)
from rl_capstone import config
