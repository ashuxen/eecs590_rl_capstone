"""
============================================================================
MDP MODULE - Markov Decision Process Classes
============================================================================

This module contains the core MDP classes used throughout the project.

FILES IN THIS MODULE:
---------------------

1. finite_mrp.py - Markov Reward Process (MRP)
   - Simpler than MDP: no actions, just follow transitions
   - Foundation for understanding value functions
   - Used in Mini 1

2. finite_mdp.py - Markov Decision Process (MDP)
   - Full RL framework: states, actions, transitions, rewards
   - Includes Value Iteration implementation
   - Used in Mini 2

3. belief_mdp.py - Belief MDP (Model-Based RL)
   - For when you DON'T know the transition probabilities!
   - Agent learns P(s'|s,a) from experience
   - Foundation for model-based RL

USAGE:
------
    from rl_capstone.mdp import FiniteMDP, BeliefMDP
    
    # Create and solve an MDP
    mdp = FiniteMDP(states, actions, P_a, R, gamma)
    V, pi, iters = mdp.value_iteration()
    
    # Or learn the MDP from experience
    belief = BeliefMDP(n_states=100, n_actions=4)
    belief.update_beliefs(s=5, a=2, r=-1, s_next=12)

============================================================================
"""

from .finite_mrp import FiniteMRP
from .finite_mdp import FiniteMDP
from .belief_mdp import BeliefMDP

__all__ = ["FiniteMRP", "FiniteMDP", "BeliefMDP"]
