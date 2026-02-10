"""
============================================================================
POLICY ITERATION - The Other Classic DP Algorithm
============================================================================

WHY THIS FILE IS IMPORTANT:
---------------------------
Policy Iteration is an alternative to Value Iteration. While Value Iteration
updates values, Policy Iteration alternates between:
1. EVALUATION: How good is my current policy?
2. IMPROVEMENT: Can I make the policy better?

KEY INSIGHT - Why Two Algorithms?
---------------------------------
Value Iteration: Updates V and extracts π at the end
Policy Iteration: Updates π directly, evaluates V for that π

WHICH IS BETTER?
- Value Iteration: Simple, but may need many iterations
- Policy Iteration: Fewer iterations, but each is more expensive
- In practice: Often similar performance

THE TWO STEPS:
--------------
1. POLICY EVALUATION:
   "Given a fixed policy π, what is V^π(s)?"
   
   V^π(s) = R(s) + γ * Σ P(s'|s,π(s)) * V^π(s')
   
   This is easier than finding V* because we don't maximize!
   We just follow the policy.

2. POLICY IMPROVEMENT:
   "Given V^π, can we find a better policy?"
   
   π'(s) = argmax_a [ R(s) + γ * Σ P(s'|s,a) * V^π(s') ]
   
   If the new policy is different, it's guaranteed to be better!
   (This is the Policy Improvement Theorem)

============================================================================
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Tuple, Any, Union


def policy_evaluation(pi: Dict[int, Any],
                      P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
                      R: np.ndarray,
                      gamma: float,
                      method: str = "iteration",
                      tol: float = 1e-10,
                      max_iter: int = 50000) -> np.ndarray:
    """
    POLICY EVALUATION: Compute V^π (value function for a fixed policy).
    
    THE QUESTION WE'RE ANSWERING:
    -----------------------------
    "If I follow policy π forever, what's the expected total reward
    starting from each state?"
    
    THE EQUATION (Bellman Expectation Equation):
    --------------------------------------------
    V^π(s) = R(s) + γ * Σ P(s'|s,π(s)) * V^π(s')
    
    Notice: No "max"! We just follow the policy.
    
    TWO METHODS TO SOLVE THIS:
    --------------------------
    1. "iteration": Repeat V <- R + γP^π V until convergence
    2. "linear": Solve (I - γP^π)V = R directly (faster but needs matrix inverse)
    
    Args:
        pi: Policy to evaluate (dict: state -> action)
        P_a: Transition matrices
        R: Reward vector
        gamma: Discount factor
        method: "iteration" or "linear"
        
    Returns:
        V: Value function V^π(s) for each state
    """
    n = len(R)
    
    if method == "linear":
        # Solve directly using linear algebra
        # This is faster but needs to build P^π matrix
        return _policy_eval_linear(pi, P_a, R, gamma)
    
    # -------------------------------------------------------------------------
    # ITERATIVE METHOD
    # -------------------------------------------------------------------------
    # Start with V = 0
    V = np.zeros(n)
    
    for iteration in range(max_iter):
        V_new = np.zeros(n)
        
        for s in range(n):
            # Get the action that policy π says to take in state s
            a = pi[s]
            
            # Get transition probabilities for that action
            Pa = P_a[a]
            
            # Compute expected future value
            if sp.issparse(Pa):
                expected_V = float(Pa.getrow(s).dot(V).item())
            else:
                expected_V = float(Pa[s, :] @ V)
            
            # V^π(s) = R(s) + γ * expected future value
            V_new[s] = R[s] + gamma * expected_V
        
        # Check convergence
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    
    return V


def _policy_eval_linear(pi, P_a, R, gamma):
    """
    Solve policy evaluation using linear algebra.
    
    THE MATH:
    ---------
    V^π = R + γ P^π V^π
    V^π - γ P^π V^π = R
    (I - γ P^π) V^π = R
    V^π = (I - γ P^π)^{-1} R
    
    This is a system of linear equations! Solve with standard methods.
    """
    n = len(R)
    
    # Build P^π: the transition matrix when following policy π
    # P^π[s, s'] = P(s'|s, π(s))
    rows, cols, data = [], [], []
    for s in range(n):
        a = pi[s]
        Pa = P_a[a]
        if sp.issparse(Pa):
            Pa_row = Pa.getrow(s)
            for idx, val in zip(Pa_row.indices, Pa_row.data):
                rows.append(s)
                cols.append(int(idx))
                data.append(float(val))
        else:
            for s_next in range(n):
                if Pa[s, s_next] > 0:
                    rows.append(s)
                    cols.append(s_next)
                    data.append(float(Pa[s, s_next]))
    
    P_pi = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Solve (I - γP^π)V = R
    A = sp.eye(n, format='csr') - gamma * P_pi
    V = spla.spsolve(A, R)
    
    return np.asarray(V)


def policy_improvement(V: np.ndarray,
                       P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
                       R: np.ndarray,
                       gamma: float,
                       actions: list) -> Dict[int, Any]:
    """
    POLICY IMPROVEMENT: Get greedy policy w.r.t. V.
    
    THE QUESTION WE'RE ANSWERING:
    -----------------------------
    "Given that I know V^π, what's the BEST action to take in each state?"
    
    THE EQUATION:
    -------------
    π'(s) = argmax_a [ R(s) + γ * Σ P(s'|s,a) * V(s') ]
    
    POLICY IMPROVEMENT THEOREM:
    ---------------------------
    If π' ≠ π, then π' is strictly better than π!
    V^{π'}(s) ≥ V^π(s) for all s
    
    This guarantees we make progress at each step.
    
    Args:
        V: Current value function
        P_a: Transition matrices
        R: Reward vector
        gamma: Discount factor
        actions: List of available actions
        
    Returns:
        pi: Improved policy (dict: state -> action)
    """
    n = len(R)
    pi = {}
    
    for s in range(n):
        best_action = None
        best_q = -np.inf
        
        # Try each action, pick the best one
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
    
    return pi


def policy_iteration(P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
                     R: np.ndarray,
                     gamma: float,
                     actions: list,
                     eval_method: str = "linear",
                     tol: float = 1e-10,
                     max_iter: int = 1000) -> Tuple[np.ndarray, Dict[int, Any], int]:
    """
    POLICY ITERATION ALGORITHM
    
    Alternates between evaluation and improvement until convergence.
    
    THE ALGORITHM:
    --------------
    1. INITIALIZE: Start with arbitrary policy (e.g., always take first action)
    
    2. REPEAT:
       a. EVALUATE: Compute V^π (how good is current policy?)
       b. IMPROVE: π' = greedy policy w.r.t. V^π
       c. If π' == π, we're done! (policy is optimal)
       d. Otherwise: π = π', go back to step a
    
    WHY IT CONVERGES:
    -----------------
    - Each improvement makes the policy better (or the same)
    - There are only finitely many policies
    - So we must eventually stop improving!
    
    CONVERGENCE GUARANTEE:
    ----------------------
    Policy Iteration converges in at most |A|^|S| iterations
    (but usually MUCH faster - often just a few iterations!)
    
    Args:
        P_a: Transition matrices
        R: Reward vector
        gamma: Discount factor
        actions: List of actions
        eval_method: "linear" (fast) or "iteration" (simple)
        
    Returns:
        V: Optimal value function V*
        pi: Optimal policy π*
        iterations: Number of policy improvement steps
    """
    n = len(R)
    
    # -------------------------------------------------------------------------
    # STEP 1: INITIALIZE with arbitrary policy
    # -------------------------------------------------------------------------
    # Start by always taking the first action
    pi = {s: actions[0] for s in range(n)}
    
    print("Starting Policy Iteration...")
    
    # -------------------------------------------------------------------------
    # STEP 2: ITERATE until policy stops changing
    # -------------------------------------------------------------------------
    for iteration in range(max_iter):
        # (a) POLICY EVALUATION: How good is our current policy?
        V = policy_evaluation(pi, P_a, R, gamma, method=eval_method, tol=tol)
        
        # (b) POLICY IMPROVEMENT: Can we do better?
        pi_new = policy_improvement(V, P_a, R, gamma, actions)
        
        # (c) Check if policy changed
        policy_stable = all(pi[s] == pi_new[s] for s in range(n))
        
        if policy_stable:
            # Policy didn't change = we found the optimal policy!
            print(f"  Policy Iteration converged in {iteration + 1} iterations")
            return V, pi, iteration + 1
        
        # (d) Policy changed, continue with new policy
        pi = pi_new
    
    # If we get here, we hit max_iter (shouldn't happen for finite MDPs)
    print(f"  Warning: Policy Iteration did not converge in {max_iter} iterations")
    V = policy_evaluation(pi, P_a, R, gamma, method=eval_method, tol=tol)
    return V, pi, max_iter


# ===================================================================================================================
# Q-VALUE VERSION OF POLICY EVALUATION (for completeness i am keeping here but I have separate Q value algo file too)
# ===================================================================================================================

def q_policy_evaluation(pi: Dict[int, Any],
                        P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]],
                        R: np.ndarray,
                        gamma: float,
                        actions: list,
                        tol: float = 1e-10,
                        max_iter: int = 50000) -> np.ndarray:
    """
    Compute Q^π(s,a) for all state-action pairs under policy π.
    
    Q^π(s,a) = R(s) + γ * Σ P(s'|s,a) * Q^π(s', π(s'))
    
    This gives us more information than just V^π - we know the value
    of each action, not just the policy's action.
    
    Returns:
        Q: Array of shape (n_states, n_actions)
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
                if sp.issparse(Pa):
                    Pa_row = Pa.getrow(s).toarray().flatten()
                else:
                    Pa_row = Pa[s, :]
                
                # Expected Q at next states, following policy π
                Q_pi_next = np.array([Q[s_next, action_idx[pi[s_next]]] for s_next in range(n)])
                expected_Q = np.dot(Pa_row, Q_pi_next)
                
                Q_new[s, a_idx] = R[s] + gamma * expected_Q
        
        if np.max(np.abs(Q_new - Q)) < tol:
            break
        Q = Q_new
    
    return Q
