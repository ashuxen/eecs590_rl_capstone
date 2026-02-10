"""
============================================================================
VALUE ITERATION - The Classic Dynamic Programming Algorithm
============================================================================

WHY THIS FILE IS IMPORTANT:
---------------------------
Value Iteration is one of the two fundamental algorithms for solving MDPs
(the other is Policy Iteration). If we understand this, we understand
the core of Dynamic Programming in RL!

THE BIG IDEA:
-------------
We want to find V*(s) = the maximum expected reward starting from state s.

But V*(s) depends on V*(s') for all next states!
This seems circular... how do we solve it?

ANSWER: We iterate!
1. Start with a guess (V = 0 everywhere)
2. Update V using the Bellman equation
3. Repeat until V stops changing
4. The fixed point IS the optimal value function!

THE BELLMAN OPTIMALITY EQUATION:
--------------------------------
V*(s) = max_a [ R(s) + γ * Σ P(s'|s,a) * V*(s') ]
        ^^^^   ^^^^   ^   ^^^^^^^^^^^^^^^^^^^^^^^^
        |      |      |   |
        |      |      |   Expected future value if we take action a
        |      |      Discount factor
        |      Immediate reward
        Best action (we pick the maximum)

A simple explanation:
---------------------
"The value of a state equals the immediate reward plus the 
discounted value of where we expect to end up, assuming we
pick the best possible action."

============================================================================
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Any, Union


def value_iteration(P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
                    R: np.ndarray,
                    gamma: float,
                    actions: list,
                    tol: float = 1e-10,
                    max_iter: int = 50000) -> Tuple[np.ndarray, Dict[int, Any], int]:
    """
    VALUE ITERATION ALGORITHM
    
    Finds the optimal value function V* and optimal policy π*.
    
    THE ALGORITHM STEP BY STEP:
    ---------------------------
    
    INITIALIZE:
        V(s) = 0 for all states (we start knowing nothing)
    
    REPEAT:
        For each state s:
            For each action a:
                Q(s,a) = R(s) + γ * Σ P(s'|s,a) * V(s')
            V_new(s) = max_a Q(s,a)
        
        If max|V_new - V| < tolerance:
            We've converged! Stop.
        
        V = V_new
    
    EXTRACT POLICY:
        For each state s:
            π(s) = action that achieves max Q(s,a)
    
    
    Args:
        P_a: Dictionary of transition matrices
             P_a[action][s, s'] = probability of going from s to s' with that action
        R: Reward vector (R[s] = reward in state s)
        gamma: Discount factor (0 to 1, typically 0.99)
        actions: List of available actions
        tol: Stop when max change is below this
        max_iter: Safety limit on iterations
        
    Returns:
        V: Optimal value function (array of length n_states)
        pi: Optimal policy (dict mapping state -> best action)
        iterations: Number of iterations until convergence
    
    
    EXAMPLE:
    --------
    Windy Chasm with B=0.5, γ=0.99:
    - Converges in ~900 iterations
    - V*(start) ≈ 7.82
    - Policy: mostly "Forward" with some corrections near walls
    """
    
    # -------------------------------------------------------------------------
    # STEP 1: INITIALIZATION
    # -------------------------------------------------------------------------
    n = len(R)  # Number of states
    V = np.zeros(n)  # Start with V(s) = 0 for all states
    
    # -------------------------------------------------------------------------
    # STEP 2: ITERATE UNTIL CONVERGENCE
    # -------------------------------------------------------------------------
    for iteration in range(max_iter):
        V_new = np.zeros(n)  # Will store updated values
        
        # For each state, find the best action
        for s in range(n):
            q_values = []  # Q(s,a) for each action
            
            # Compute Q(s,a) for each action
            for a in actions:
                # Get transition probabilities P(s'|s,a)
                Pa = P_a[a]
                
                # Compute expected future value: Σ P(s'|s,a) * V(s')
                # This is the "where will I end up, and how good is that?" part
                if sp.issparse(Pa):
                    # Sparse matrix version (efficient for large state spaces)
                    expected_V = float(Pa.getrow(s).dot(V).item())
                else:
                    # Dense matrix version
                    expected_V = float(Pa[s, :] @ V)
                
                # Q(s,a) = R(s) + γ * expected_V
                # "Immediate reward + discounted future value"
                q_sa = R[s] + gamma * expected_V
                q_values.append(q_sa)
            
            # V(s) = max over all actions
            # "The value of s is the best Q-value we can achieve"
            V_new[s] = max(q_values)
        
        # Check for convergence
        # If V didn't change much, we've found the fixed point!
        delta = np.max(np.abs(V_new - V))
        
        if delta < tol:
            print(f"  Value Iteration converged in {iteration + 1} iterations")
            print(f"  Final max change: {delta:.2e}")
            break
        
        # Update V for next iteration
        V = V_new
    
    # -------------------------------------------------------------------------
    # STEP 3: EXTRACT OPTIMAL POLICY
    # -------------------------------------------------------------------------
    # Now that we have V*, find the best action for each state
    pi = {}
    
    for s in range(n):
        best_action = None
        best_q = -np.inf
        
        for a in actions:
            Pa = P_a[a]
            if sp.issparse(Pa):
                expected_V = float(Pa.getrow(s).dot(V).item())
            else:
                expected_V = float(Pa[s, :] @ V)
            
            q_sa = R[s] + gamma * expected_V
            
            if q_sa > best_q:
                best_q = q_sa
                best_action = a
        
        pi[s] = best_action
    
    return V, pi, iteration + 1


# =============================================================================
# VERBOSE VERSION (with logging for debugging/visualization)
# =============================================================================

def value_iteration_verbose(P_a, R, gamma, actions, 
                            tol=1e-10, max_iter=50000, 
                            log_interval=100):
    """
    Same as value_iteration but saves history for visualization.
    
    Returns extra info:
    - deltas: How much V changed each iteration (for plotting convergence)
    - v_history: Snapshots of V at regular intervals
    """
    n = len(R)
    V = np.zeros(n)
    deltas = []
    v_history = []
    
    for iteration in range(max_iter):
        V_new = np.zeros(n)
        
        for s in range(n):
            q_values = []
            for a in actions:
                Pa = P_a[a]
                if sp.issparse(Pa):
                    expected_V = float(Pa.getrow(s).dot(V).item())
                else:
                    expected_V = float(Pa[s, :] @ V)
                q_sa = R[s] + gamma * expected_V
                q_values.append(q_sa)
            V_new[s] = max(q_values)
        
        delta = np.max(np.abs(V_new - V))
        deltas.append(delta)
        
        # Save snapshot periodically
        if iteration % log_interval == 0:
            v_history.append(V.copy())
        
        if delta < tol:
            break
        V = V_new
    
    # Extract policy
    pi = {}
    for s in range(n):
        best_a, best_q = None, -np.inf
        for a in actions:
            Pa = P_a[a]
            if sp.issparse(Pa):
                expected_V = float(Pa.getrow(s).dot(V).item())
            else:
                expected_V = float(Pa[s, :] @ V)
            q_sa = R[s] + gamma * expected_V
            if q_sa > best_q:
                best_q, best_a = q_sa, a
        pi[s] = best_a
    
    info = {
        "iterations": iteration + 1,
        "converged": delta < tol,
        "final_delta": delta,
        "deltas": np.array(deltas),
        "v_history": v_history
    }
    
    return V, pi, info
