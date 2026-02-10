"""
Q-Value Iteration Algorithm
===========================

Requirement 4: DP implementation on Q-values.

Q-Value Iteration finds optimal Q* by iterating:
    Q_{k+1}(s,a) = R(s,a) + γ Σ P(s'|s,a) max_{a'} Q_k(s',a')

Policy is then: π*(s) = argmax_a Q*(s,a)
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Any, Union


def q_value_iteration(P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
                      R: np.ndarray,
                      gamma: float,
                      actions: list,
                      tol: float = 1e-10,
                      max_iter: int = 50000) -> Tuple[np.ndarray, Dict[int, Any], int]:
    """
    Q-Value Iteration for MDPs.
    
    Finds optimal Q-function Q*(s,a) and optimal policy π*.
    
    Args:
        P_a: Dict mapping action -> transition matrix P(s'|s,a)
        R: Reward vector R(s) of length n  
        gamma: Discount factor
        actions: List of actions
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Q: Optimal Q-values, shape (n_states, n_actions)
        pi: Optimal policy (dict: state -> action)
        iterations: Number of iterations
    """
    n = len(R)
    n_actions = len(actions)
    action_idx = {a: i for i, a in enumerate(actions)}
    
    Q = np.zeros((n, n_actions))
    
    for iteration in range(max_iter):
        Q_new = np.zeros_like(Q)
        
        for s in range(n):
            for a_idx, a in enumerate(actions):
                Pa = P_a[a]
                
                # Get transition probabilities from state s
                if sp.issparse(Pa):
                    Pa_row = Pa.getrow(s).toarray().flatten()
                else:
                    Pa_row = Pa[s, :]
                
                # Expected max Q at next states
                max_Q_next = np.max(Q, axis=1)  # max over actions for each state
                expected_max_Q = np.dot(Pa_row, max_Q_next)
                
                Q_new[s, a_idx] = R[s] + gamma * expected_max_Q
        
        delta = np.max(np.abs(Q_new - Q))
        if delta < tol:
            break
        Q = Q_new
    
    # Extract greedy policy
    pi = {}
    for s in range(n):
        best_a_idx = int(np.argmax(Q[s, :]))
        pi[s] = actions[best_a_idx]
    
    return Q, pi, iteration + 1


def q_from_v(V: np.ndarray,
             P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
             R: np.ndarray,
             gamma: float,
             actions: list) -> np.ndarray:
    """
    Compute Q-values from value function V.
    
    Q(s,a) = R(s) + γ Σ P(s'|s,a) V(s')
    
    Args:
        V: Value function
        P_a: Transition matrices
        R: Reward vector
        gamma: Discount factor
        actions: Action list
        
    Returns:
        Q: Q-values, shape (n_states, n_actions)
    """
    n = len(R)
    n_actions = len(actions)
    Q = np.zeros((n, n_actions))
    
    for s in range(n):
        for a_idx, a in enumerate(actions):
            Pa = P_a[a]
            if sp.issparse(Pa):
                expected_V = float(Pa.getrow(s).dot(V).item())
            else:
                expected_V = float(Pa[s, :] @ V)
            Q[s, a_idx] = R[s] + gamma * expected_V
    
    return Q
