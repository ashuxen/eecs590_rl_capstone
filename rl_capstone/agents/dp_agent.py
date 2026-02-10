"""
============================================================================
DYNAMIC PROGRAMMING AGENT - Learns Optimal Policy via DP
============================================================================

Author: Ashutosh Kumar

WHY THIS FILE IS IMPORTANT:
---------------------------
This is where theory meets practice! The DPAgent:
1. Takes an environment
2. Solves it using Dynamic Programming
3. Stores the optimal policy and value function
4. Can select actions and be evaluated

This is the "brain" of our RL system - it learns and makes decisions.

HOW IT CONNECTS TO OTHER FILES:
-------------------------------
- Uses `FiniteMDP` from mdp/finite_mdp.py
- Uses algorithms from algorithms/value_iteration.py
- Is managed by `Trainer` from trainer.py
- Interacts with `WindyChasmEnv` from environments/

============================================================================
"""

import numpy as np
import pickle
from typing import Any, Dict, Optional

from .base_agent import BaseAgent
from ..algorithms import value_iteration, q_value_iteration, policy_iteration


class DPAgent(BaseAgent):
    """
    An agent that uses Dynamic Programming to find optimal behavior.
    
    WHAT THIS AGENT DOES:
    ---------------------
    1. Takes an environment (like Windy Chasm)
    2. Extracts the MDP structure from it
    3. Runs a DP algorithm (Value Iteration, Policy Iteration, etc.)
    4. Stores the resulting V* and π*
    5. Uses π* to select actions during evaluation
    
    SUPPORTED METHODS:
    ------------------
    - "value_iteration": Find V* first, then extract π*
    - "q_value_iteration": Find Q* first, then extract π*
    - "policy_iteration": Alternate between evaluation and improvement
    
    WHY MULTIPLE METHODS?
    ---------------------
    They all find the optimal policy, but:
    - Value Iteration: Simple, reliable
    - Q-Value Iteration: Gives Q-values (useful for some applications)
    - Policy Iteration: Often fewer iterations, but each is expensive
    
    Usage:
        env = WindyChasmEnv(B=0.5)
        agent = DPAgent(env, method="value_iteration")
        agent.train()
        
        # Now agent can select optimal actions
        action = agent.select_action(state=42)
    """
    
    def __init__(self, env, gamma: float = 0.99,
                 method: str = "value_iteration",
                 tol: float = 1e-10,
                 max_iter: int = 50000):
        """
        Create a DP agent.
        
        Args:
            env: Environment to solve (must have get_mdp() method)
            gamma: Discount factor (usually 0.99)
            method: Which DP algorithm to use
            tol: Convergence tolerance
            max_iter: Safety limit on iterations
        """
        # Call parent class __init__
        super().__init__(env, gamma)
        
        # Store hyperparameters
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        
        # Get environment dimensions
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # Get action names (e.g., ["F", "L", "R"])
        if hasattr(env, 'actions'):
            self.actions = env.actions
        else:
            self.actions = list(range(env.n_actions))
    
    def select_action(self, state: int) -> Any:
        """
        Select the optimal action for a given state.
        
        This is what makes the agent "smart" - it uses the learned
        policy to make decisions!
        
        Args:
            state: State index (e.g., 42)
            
        Returns:
            action: The best action according to π*
        """
        if self.pi is None:
            raise ValueError("Agent not trained! Call train() first.")
        
        # Look up the optimal action for this state
        return self.pi[state]
    
    def train(self) -> Dict[str, Any]:
        """
        Train the agent by solving the MDP.
        
        This is where the magic happens! We run the DP algorithm
        and store the resulting policy.
        
        Returns:
            info: Dictionary with training results:
                  - method: Which algorithm was used
                  - iterations: How many iterations it took
                  - v_start: Value at starting state
        """
        print(f"\n{'='*50}")
        print(f"Training DPAgent with {self.method}")
        print(f"{'='*50}")
        
        # Get MDP from environment
        mdp = self.env.get_mdp()
        P_a = mdp.P_a      # Transition matrices
        R = mdp.R_s        # Reward vector
        
        # Run the appropriate algorithm
        if self.method == "value_iteration":
            # Value Iteration: Find V*, then extract π*
            self.V, pi_dict, iters = value_iteration(
                P_a, R, self.gamma, self.actions, self.tol, self.max_iter
            )
            self.pi = pi_dict
            
        elif self.method == "q_value_iteration":
            # Q-Value Iteration: Find Q*, then extract π*
            self.Q, pi_dict, iters = q_value_iteration(
                P_a, R, self.gamma, self.actions, self.tol, self.max_iter
            )
            self.pi = pi_dict
            # Also compute V from Q for convenience
            self.V = np.max(self.Q, axis=1)
            
        elif self.method == "policy_iteration":
            # Policy Iteration: Alternate evaluation and improvement
            self.V, pi_dict, iters = policy_iteration(
                P_a, R, self.gamma, self.actions, tol=self.tol, max_iter=self.max_iter
            )
            self.pi = pi_dict
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Mark as trained
        self.is_trained = True
        
        # Get value at start state for reporting
        start_state = getattr(self.env, 'start_state_idx', 0)
        v_start = float(self.V[start_state])
        
        # Store training info
        self.training_info = {
            "method": self.method,
            "iterations": iters,
            "v_start": v_start,
            "v_mean": float(np.mean(self.V)),
            "v_max": float(np.max(self.V)),
            "v_min": float(np.min(self.V)),
            "gamma": self.gamma
        }
        
        print(f"\nTraining complete!")
        print(f"  Iterations: {iters}")
        print(f"  V*(start) = {v_start:.4f}")
        
        return self.training_info
    
    def save(self, filepath: str) -> None:
        """
        Save the trained agent to a file.
        
        This saves the "policy kernel" - the learned V* and π*.
        You can load this later without re-training!
        
        The file contains:
        - V: Value function
        - Q: Q-values (if computed)
        - pi: Policy
        - All hyperparameters
        
        Args:
            filepath: Where to save (e.g., "models/policy_kernel/v1.pkl")
        """
        data = {
            # Learned values
            "V": self.V,
            "Q": self.Q,
            "pi": self.pi,
            
            # Hyperparameters
            "gamma": self.gamma,
            "method": self.method,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "actions": self.actions,
            
            # Training info
            "training_info": self.training_info,
            "is_trained": self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved agent to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, env) -> "DPAgent":
        """
        Load a previously saved agent.
        
        Args:
            filepath: Where to load from
            env: Environment (needed to create agent)
            
        Returns:
            agent: The loaded agent, ready to use!
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new agent with saved hyperparameters
        agent = cls(env, gamma=data["gamma"], method=data["method"])
        
        # Restore learned values
        agent.V = data["V"]
        agent.Q = data["Q"]
        agent.pi = data["pi"]
        agent.training_info = data["training_info"]
        agent.is_trained = data["is_trained"]
        
        print(f"Loaded agent from {filepath}")
        return agent
    
    def print_policy_grid(self, grid_i: int = 20, grid_j: int = 7):
        """
        Print the policy as a visual grid.
        
        This helps you see what the agent learned!
        
        Example output:
            j=6| -> -> -> ... G
            j=5| -> -> -> ... G
            j=4| -> -> -> ... G
            j=3| -> -> -> ... G  (start)
            j=2| -> -> -> ... G
            j=1| -> -> -> ... G
            j=0| -> -> -> ... G
        """
        if self.pi is None:
            print("Agent not trained!")
            return
        
        action_symbols = {"F": "->", "L": "v ", "R": "^ ", 0: "->", 1: "v ", 2: "^ "}
        
        print("\nOptimal Policy:")
        for j in range(grid_j - 1, -1, -1):
            row = f"j={j}|"
            for i in range(grid_i):
                # Find state index for (i, j)
                s_idx = i * grid_j + j  # Assumes standard ordering
                if s_idx in self.pi and s_idx < self.n_states - 2:
                    action = self.pi[s_idx]
                    row += action_symbols.get(action, "? ") + " "
                else:
                    row += "   "
            if j == 3:
                row += " <- START"
            print(row)
        print("-" * 65)
