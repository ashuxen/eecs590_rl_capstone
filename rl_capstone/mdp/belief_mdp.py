"""
============================================================================
BELIEF MDP - Learning the World Model (Model-Based RL)
============================================================================

WHY THIS FILE IS IMPORTANT:
---------------------------
In the real world, we don't know the transition probabilities P(s'|s,a)!
A robot doesn't know exactly what happens when it moves forward.
This file shows how to LEARN the MDP from experience.

This is the foundation of MODEL-BASED REINFORCEMENT LEARNING:
- The agent observes transitions (s, a) -> s', r
- It builds a mental model of how the world works
- Then it plans using this learned model

REAL WORLD EXAMPLE:
-------------------
Imagine learning to ride a bike:
- We don't know physics equations for balance
- We try actions and observe what happens
- Our brain builds a model: "if I lean left, bike goes left"
- We use this model to plan our next move

THIS CLASS DOES THE SAME THING:
- Observe: (current_state, action) -> (reward, next_state)
- Update beliefs about P(s'|s,a) and R(s,a)
- Use beliefs to plan (value iteration on believed model)

============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import json


@dataclass
class BeliefMDP:
    """
    An MDP where the agent LEARNS the transition and reward functions.
    
    Instead of knowing P(s'|s,a) exactly, the agent:
    1. Starts with uniform beliefs (anything could happen)
    2. Observes transitions from experience
    3. Updates beliefs based on what it sees
    4. Plans using current beliefs
    
    This is how real agents learn about unknown environments!
    
    Attributes:
        n_states: Number of states in the environment
        n_actions: Number of possible actions
        gamma: Discount factor for planning
        prior_count: Initial "pseudocounts" (smoothing parameter)
                     Higher = more exploration before trusting observations
    """
    
    n_states: int
    n_actions: int
    gamma: float = 0.99
    prior_count: float = 1.0  # Start with this many "imaginary" visits per state
    
    # These store what we've learned (initialized in __post_init__)
    transition_counts: np.ndarray = field(default=None, repr=False)
    reward_sum: np.ndarray = field(default=None, repr=False)
    reward_count: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """
        Initialize our belief structures.
        
        We use COUNTS instead of probabilities because it's easier to update:
        - See transition s->s' with action a? Add 1 to count[s,a,s']
        - Want probability? Divide by total counts from (s,a)
        
        This is related to BAYESIAN LEARNING with Dirichlet priors!
        """
        n, m = self.n_states, self.n_actions
        
        # transition_counts[s, a, s'] = how many times we saw (s,a)->s'
        # Start with prior_count everywhere (uniform prior belief)
        self.transition_counts = np.ones((n, m, n)) * self.prior_count
        
        # For rewards, track sum and count to compute average
        self.reward_sum = np.zeros((n, m, n))
        self.reward_count = np.ones((n, m, n)) * 0.01  # Small prior to avoid divide-by-zero
    
    # =========================================================================
    # LEARNING FROM EXPERIENCE
    # =========================================================================
    
    def update_beliefs(self, s: int, a: int, r: float, s_next: int) -> None:
        """
        Learn from a single experience: (s, a) -> r, s'
        
        This is the CORE LEARNING STEP. Every time the agent takes an action
        and sees what happens, it calls this function to update its beliefs.
        
        WHAT HAPPENS:
        -------------
        1. We were in state s
        2. We took action a
        3. We got reward r
        4. We ended up in state s_next
        
        NOW WE UPDATE OUR BELIEFS:
        - "When I do action a in state s, I sometimes end up at s_next"
        - "When (s,a)->s_next happens, I get reward r"
        
        Args:
            s: State we were in
            a: Action we took
            r: Reward we received
            s_next: State we ended up in
        """
        # Update transition count: we saw one more (s,a)->s_next transition
        self.transition_counts[s, a, s_next] += 1
        
        # Update reward estimate: add this reward to our running sum
        self.reward_sum[s, a, s_next] += r
        self.reward_count[s, a, s_next] += 1
    
    def update_beliefs_batch(self, transitions: list) -> None:
        """
        Learn from multiple experiences at once.
        
        Useful when you have a batch of recorded experiences.
        
        Args:
            transitions: List of (s, a, r, s_next) tuples
        """
        for s, a, r, s_next in transitions:
            self.update_beliefs(s, a, r, s_next)
    
    # =========================================================================
    # QUERYING OUR BELIEFS
    # =========================================================================
    
    def get_believed_transition(self, s: int, a: int) -> np.ndarray:
        """
        What do we BELIEVE happens when we do action a in state s?
        
        Returns P_belief(s'|s,a) - our current estimate of transition probs.
        
        HOW IT WORKS:
        -------------
        P(s'|s,a) ≈ count(s,a,s') / total_count(s,a)
        
        This is the MAXIMUM LIKELIHOOD ESTIMATE with Dirichlet smoothing.
        
        Example:
            If we did action "forward" in state 5 a total of 100 times:
            - 70 times we went to state 6
            - 20 times we went to state 5 (didn't move)
            - 10 times we went to state 7 (wind pushed us)
            
            Then: P(6|5,forward) ≈ 0.70
                  P(5|5,forward) ≈ 0.20
                  P(7|5,forward) ≈ 0.10
        
        Returns:
            Array of length n_states with P(s'|s,a) for each possible s'
        """
        counts = self.transition_counts[s, a, :]
        return counts / counts.sum()  # Normalize to get probabilities
    
    def get_believed_reward(self, s: int, a: int, s_next: Optional[int] = None) -> float:
        """
        What reward do we expect for (s, a) or (s, a, s')?
        
        Returns our current estimate of the reward.
        
        HOW IT WORKS:
        -------------
        R_belief = average of observed rewards = sum / count
        """
        if s_next is not None:
            # Specific next state: return average reward for (s,a,s')
            if self.reward_count[s, a, s_next] > 0.01:
                return self.reward_sum[s, a, s_next] / self.reward_count[s, a, s_next]
            return 0.0
        
        # No specific next state: return expected reward over all possibilities
        P_s_a = self.get_believed_transition(s, a)
        expected_r = 0.0
        for s_next in range(self.n_states):
            if self.reward_count[s, a, s_next] > 0.01:
                r_avg = self.reward_sum[s, a, s_next] / self.reward_count[s, a, s_next]
                expected_r += P_s_a[s_next] * r_avg
        return expected_r
    
    # =========================================================================
    # PLANNING WITH BELIEFS
    # =========================================================================
    
    def plan_with_beliefs(self, tol: float = 1e-6, max_iter: int = 1000) -> Tuple[np.ndarray, dict]:
        """
        Use our learned model to find the best policy.
        
        This is MODEL-BASED PLANNING:
        1. We've learned what we THINK the world does (from experience)
        2. Now we do value iteration on our BELIEVED model
        3. The resulting policy is our best guess at optimal behavior
        
        WHY THIS IS POWERFUL:
        ---------------------
        - We can plan without more environment interaction
        - We can "imagine" what would happen with different actions
        - This is how humans plan: we simulate outcomes in our heads!
        
        Returns:
            V: Value function under our believed model
            pi: Policy (dict: state -> action)
        """
        V = np.zeros(self.n_states)
        
        # Value iteration on believed model
        for iteration in range(max_iter):
            V_new = np.zeros(self.n_states)
            
            for s in range(self.n_states):
                # Try each action, compute Q-value using BELIEVED model
                q_values = []
                for a in range(self.n_actions):
                    P_sa = self.get_believed_transition(s, a)  # Our belief about transitions
                    R_sa = self.get_believed_reward(s, a)      # Our belief about rewards
                    
                    # Q(s,a) = R + γ * expected future value
                    q = R_sa + self.gamma * np.dot(P_sa, V)
                    q_values.append(q)
                
                V_new[s] = max(q_values)  # Best action's value
            
            # Check convergence
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        
        # Extract greedy policy from V
        pi = {}
        for s in range(self.n_states):
            best_a, best_q = 0, -np.inf
            for a in range(self.n_actions):
                P_sa = self.get_believed_transition(s, a)
                R_sa = self.get_believed_reward(s, a)
                q = R_sa + self.gamma * np.dot(P_sa, V)
                if q > best_q:
                    best_q, best_a = q, a
            pi[s] = best_a
        
        return V, pi
    
    # =========================================================================
    # EXPLORATION BONUS
    # =========================================================================
    
    def get_uncertainty(self, s: int, a: int) -> float:
        """
        How uncertain are we about (s, a)?
        
        Uses ENTROPY to measure uncertainty:
        - High entropy = very uncertain, should explore more
        - Low entropy = confident, can exploit
        
        This is useful for EXPLORATION strategies like UCB!
        """
        P = self.get_believed_transition(s, a)
        P_safe = np.clip(P, 1e-10, 1.0)  # Avoid log(0)
        entropy = -np.sum(P_safe * np.log(P_safe))
        return entropy
    
    # =========================================================================
    # SAVING AND LOADING
    # =========================================================================
    
    def save(self, filepath: str) -> None:
        """Save beliefs to a file (so we can continue learning later)."""
        data = {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "transition_counts": self.transition_counts.tolist(),
            "reward_sum": self.reward_sum.tolist(),
            "reward_count": self.reward_count.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Saved beliefs to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "BeliefMDP":
        """Load beliefs from a file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        belief = cls(
            n_states=data["n_states"],
            n_actions=data["n_actions"],
            gamma=data["gamma"]
        )
        belief.transition_counts = np.array(data["transition_counts"])
        belief.reward_sum = np.array(data["reward_sum"])
        belief.reward_count = np.array(data["reward_count"])
        return belief


# =============================================================================
# An simple example of Learning a simple environment
# =============================================================================
"""
# Create a belief MDP for a 5-state, 2-action environment
belief = BeliefMDP(n_states=5, n_actions=2, gamma=0.99)

# Simulate some experiences
experiences = [
    (0, 0, -1, 1),  # State 0, action 0 -> reward -1, state 1
    (1, 0, -1, 2),  # State 1, action 0 -> reward -1, state 2
    (2, 0, -1, 3),
    (3, 0, +10, 4), # Reached goal!
    (0, 0, -1, 1),  # More experience...
    (1, 0, -1, 2),
]

# Learn from experiences
belief.update_beliefs_batch(experiences)

# Check what we learned
print("P(s'|s=0, a=0):", belief.get_believed_transition(0, 0))

# Plan using learned model
V, pi = belief.plan_with_beliefs()
print("Learned value function:", V)
print("Learned policy:", pi)
"""
