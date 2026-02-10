"""
============================================================================
FINITE MARKOV DECISION PROCESS (MDP) - Core Class
============================================================================

WHY THIS FILE IS IMPORTANT:
---------------------------
This is the foundation of everything in Reinforcement Learning! An MDP is
the mathematical framework that defines how an agent interacts with an
environment. Every RL problem can be described as an MDP.

WHAT IS AN MDP?
---------------
An MDP is defined by 5 things (remember: S, A, P, R, γ):

    S = State space      : All possible situations the agent can be in
    A = Action space     : All possible actions the agent can take
    P = Transitions      : P(s'|s,a) = probability of going to state s' 
                           if you're in state s and take action a
    R = Rewards          : R(s) = reward you get in state s
    γ = Discount factor  : How much we care about future rewards (0 to 1)
                           γ=0.99 means future rewards are almost as good as now
                           γ=0.5 means future rewards are worth half

EXAMPLE (Windy Chasm from Mini 2):
----------------------------------
    S = Grid positions (0,0) to (19,6) + crash + goal = 142 states
    A = {Forward, Left, Right} = 3 actions
    P = Wind pushes you randomly toward walls
    R = -1 per step, +20 for goal, -5 for crash
    γ = 0.99

============================================================================
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Union


@dataclass
class FiniteMDP:
    """
    A Finite Markov Decision Process.
    
    This class stores the MDP definition and provides methods to solve it.
    
    Attributes (the 5 components of an MDP):
        states  : List of all states (e.g., [(0,0), (0,1), ..., "crash", "goal"])
        actions : List of all actions (e.g., ["Forward", "Left", "Right"])
        P_a     : Transition probabilities - a dictionary where:
                  P_a["Forward"][s, s'] = probability of going from s to s' with action "Forward"
        R_s     : Reward for each state - a numpy array where R_s[s] = reward in state s
        gamma   : Discount factor (how much we value future rewards)
    """
    
    # These are the 5 components of an MDP
    states: List[Any]                                    # S: state space
    actions: List[Any]                                   # A: action space
    P_a: Dict[Any, Union[np.ndarray, sp.csr_matrix]]    # P: transition probabilities
    R_s: np.ndarray                                      # R: rewards
    gamma: float                                         # γ: discount factor
    
    # These store our solutions (computed later)
    _V_star: Optional[np.ndarray] = field(default=None, repr=False)
    _Q_star: Optional[np.ndarray] = field(default=None, repr=False)
    _pi_star: Optional[Dict[int, Dict[Any, float]]] = field(default=None, repr=False)
    
    def __post_init__(self):
        """
        This runs automatically after creating an MDP.
        It checks that all the pieces fit together correctly.
        """
        n = len(self.states)
        
        # Check: Each transition matrix should be n×n (can go from any state to any state)
        for a, Pa in self.P_a.items():
            assert Pa.shape == (n, n), f"Transition matrix for action {a} has wrong shape!"
        
        # Check: Reward vector should have one value per state
        assert len(self.R_s) == n, f"Reward vector has {len(self.R_s)} entries but we have {n} states!"
    
    # =========================================================================
    # BASIC PROPERTIES
    # =========================================================================
    
    @property
    def n_states(self) -> int:
        """How many states are in this MDP?"""
        return len(self.states)
    
    @property
    def n_actions(self) -> int:
        """How many actions can the agent take?"""
        return len(self.actions)
    
    # =========================================================================
    # Q-VALUE COMPUTATION
    # =========================================================================
    
    def compute_q_value(self, s: int, a: Any, V: np.ndarray) -> float:
        """
        Compute Q(s,a) - the value of taking action a in state s.
        
        THE Q-VALUE EQUATION (from lectures):
        -------------------------------------
        Q(s,a) = R(s) + γ * Σ P(s'|s,a) * V(s')
                 ^^^^   ^   ^^^^^^^^^^   ^^^^^
                 |      |   |            |
                 |      |   |            Value of next state
                 |      |   Probability of reaching s'
                 |      Discount factor
                 Immediate reward
        
        In words: "The value of taking action a in state s equals the
        immediate reward plus the discounted expected value of where we end up."
        
        Args:
            s: Current state index (which state we're in)
            a: Action to evaluate (which action we're considering)
            V: Current value function estimate (how good is each state?)
            
        Returns:
            Q(s,a): The expected total reward from taking action a in state s
        """
        # Get the transition probabilities for this action
        Pa = self.P_a[a]
        
        # Calculate expected value of next states: Σ P(s'|s,a) * V(s')
        # This is a dot product: [P(s'=0), P(s'=1), ...] · [V(0), V(1), ...]
        if sp.issparse(Pa):
            # For sparse matrices (efficient for large state spaces)
            expected_value = float(Pa.getrow(s).dot(V).item())
        else:
            # For regular matrices
            expected_value = float(Pa[s, :] @ V)
        
        # Q(s,a) = R(s) + γ * expected_value
        return self.R_s[s] + self.gamma * expected_value
    
    # =========================================================================
    # POLICY EXTRACTION
    # =========================================================================
    
    def greedy_policy_from_V(self, V: np.ndarray) -> Dict[int, Dict[Any, float]]:
        """
        Given a value function V, find the BEST action in each state.
        
        THE GREEDY POLICY (from lectures):
        ----------------------------------
        π*(s) = argmax_a Q(s,a)
        
        In words: "The best action in state s is the one with highest Q-value."
        
        This is called "greedy" because we always pick the best option,
        no exploration, no randomness.
        
        Args:
            V: Value function (how good is each state?)
            
        Returns:
            pi: Dictionary where pi[s] = {best_action: 1.0}
                (The 1.0 means we take that action with 100% probability)
        """
        pi = {}  # Our policy: will map state -> action
        
        # For each state, find the best action
        for s in range(self.n_states):
            best_action = None
            best_q_value = -np.inf  # Start with worst possible value
            
            # Try each action and see which gives highest Q-value
            for a in self.actions:
                q_sa = self.compute_q_value(s, a, V)
                
                if q_sa > best_q_value:
                    best_q_value = q_sa
                    best_action = a
            
            # Store the best action for this state
            # Format: {action: probability} - we choose best_action with prob 1.0
            pi[s] = {best_action: 1.0}
        
        return pi
    
    # =========================================================================
    # VALUE ITERATION - The main algorithm!
    # =========================================================================
    
    def value_iteration(self, tol: float = 1e-10, 
                        max_iter: int = 50000) -> Tuple[np.ndarray, Dict, int]:
        """
        VALUE ITERATION - Find the optimal value function V* and policy π*.
        
        THE BELLMAN OPTIMALITY EQUATION (the key insight!):
        ---------------------------------------------------
        V*(s) = max_a [R(s) + γ * Σ P(s'|s,a) * V*(s')]
        
        In words: "The optimal value of a state equals the best action's
        immediate reward plus discounted expected future value."
        
        THE ALGORITHM:
        --------------
        1. Start with V(s) = 0 for all states (we know nothing)
        2. Repeat until convergence:
           - For each state s:
             - Try all actions
             - V_new(s) = best Q-value we found
           - If V didn't change much, we're done!
        3. Extract policy: π(s) = action with highest Q-value
        
        WHY IT WORKS:
        -------------
        Each iteration makes V more accurate. The Bellman equation is a
        "contraction" - it's guaranteed to converge to the true V*.
        
        Args:
            tol: Stop when max change in V is less than this (convergence check)
            max_iter: Maximum iterations (safety limit)
            
        Returns:
            V: Optimal value function V*(s) for each state
            pi: Optimal policy π*(s) for each state
            iterations: How many iterations it took
        """
        # Step 1: Initialize V(s) = 0 for all states
        V = np.zeros(self.n_states)
        
        # Step 2: Keep updating until convergence
        for iteration in range(max_iter):
            V_new = np.empty_like(V)  # Will store updated values
            
            # For each state, find the best action's value
            for s in range(self.n_states):
                # Compute Q(s,a) for all actions
                q_values = [self.compute_q_value(s, a, V) for a in self.actions]
                
                # V(s) = max Q(s,a) - the best we can do from state s
                V_new[s] = max(q_values)
            
            # Check convergence: did V change much?
            max_change = np.max(np.abs(V_new - V))
            
            if max_change < tol:
                # Converged! Values stopped changing.
                break
            
            # Update V for next iteration
            V = V_new
        
        # Step 3: Extract the optimal policy from V*
        # Remember: self._V_star stores V for later use
        self._V_star = V
        self._pi_star = self.greedy_policy_from_V(V)
        
        return V, self._pi_star, iteration + 1
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    @property
    def V_star(self) -> np.ndarray:
        """
        Get the optimal value function (solve if not already done).
        
        Usage:
            mdp = FiniteMDP(...)
            print(mdp.V_star)  # Automatically solves if needed!
        """
        if self._V_star is None:
            self.value_iteration()
        return self._V_star
    
    @property
    def pi_star(self) -> Dict[int, Dict[Any, float]]:
        """Get the optimal policy (solve if not already done)."""
        if self._pi_star is None:
            self.value_iteration()
        return self._pi_star


# =============================================================================
# VERY SIMPLE EXAMPLE USAGE (we cam uncomment if needed to test)
# =============================================================================
"""
# Create a simple 3-state MDP
import numpy as np

states = ["start", "middle", "goal"]
actions = ["go"]

# Transition: start -> middle -> goal (deterministic)
P = np.array([
    [0, 1, 0],  # From start, go to middle
    [0, 0, 1],  # From middle, go to goal  
    [0, 0, 1],  # Goal is absorbing (stay there)
])

R = np.array([-1, -1, 10])  # -1 per step, +10 at goal
gamma = 0.99

mdp = FiniteMDP(states, ["go"], {"go": P}, R, gamma)
V, pi, iters = mdp.value_iteration()

print(f"Converged in {iters} iterations")
print(f"V(start) = {V[0]:.2f}")  # Should be about 8.01 (10 * 0.99^2 - 1 - 0.99)
"""
