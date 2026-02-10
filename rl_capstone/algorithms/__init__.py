"""
============================================================================
ALGORITHMS MODULE - Dynamic Programming for MDPs
============================================================================

This module contains the classic DP algorithms for solving MDPs.

FILES IN THIS MODULE:
---------------------

1. value_iteration.py - Value Iteration
   - THE classic algorithm for finding V* and π*
   - Iterates: V(s) = max_a [R(s) + γ Σ P(s'|s,a) V(s')]
   - Simple, reliable, works for any finite MDP

2. q_value_iteration.py - Q-Value Iteration  
   - Same idea but for Q-values instead of V-values
   - Iterates: Q(s,a) = R(s) + γ Σ P(s'|s,a) max_a' Q(s',a')
   - Useful when you need Q-values directly

3. policy_iteration.py - Policy Iteration
   - Alternative to Value Iteration
   - Alternates: Evaluation (compute V^π) and Improvement (make π greedy)
   - Often fewer iterations but each is more expensive

4. td_lambda.py - TD(λ) Learning
   - For when you DON'T have the model!
   - Learns from experience without knowing P(s'|s,a)
   - Bridge between DP and model-free RL

WHICH TO USE?
-------------
- Know the MDP? → Value Iteration (simplest)
- Need Q-values? → Q-Value Iteration
- Want fast convergence? → Policy Iteration (try it!)
- Don't know transitions? → TD(λ) or other model-free methods

USAGE:
------
    from rl_capstone.algorithms import value_iteration, policy_iteration
    
    # Solve an MDP
    V, pi, iters = value_iteration(P_a, R, gamma, actions)

============================================================================
"""

from .value_iteration import value_iteration, value_iteration_verbose
from .q_value_iteration import q_value_iteration
from .policy_iteration import policy_iteration, policy_evaluation, policy_improvement
from .td_lambda import td_lambda, td_zero

__all__ = [
    "value_iteration", "value_iteration_verbose",
    "q_value_iteration", 
    "policy_iteration", "policy_evaluation", "policy_improvement",
    "td_lambda", "td_zero"
]
