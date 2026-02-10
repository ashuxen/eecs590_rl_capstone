"""
============================================================================
BASE AGENT - Abstract Interface for All Agents
============================================================================

WHY THIS FILE IS IMPORTANT:
---------------------------
This defines WHAT an agent should be able to do, without saying HOW.
All specific agents (DPAgent, etc.) inherit from this class.

This is called an "abstract base class" (ABC).

WHAT EVERY AGENT MUST DO:
-------------------------
1. select_action(state) → action  : Choose what to do
2. train() → info                 : Learn from the environment
3. save(path)                     : Save to file
4. load(path) → agent             : Load from file

WHY USE INHERITANCE?
--------------------
- Forces all agents to have the same interface
- Makes it easy to swap agents without changing other code
- Documents what agents need to implement

============================================================================
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    METHODS YOU MUST IMPLEMENT (in subclasses):
    -------------------------------------------
    - select_action(state): Choose action given state
    - train(): Learn policy/value function
    - save(path): Save agent to file
    - load(path): Load agent from file
    
    METHODS PROVIDED (already implemented):
    ---------------------------------------
    - evaluate_episode(): Run one episode and return results
    - get_value_function(): Return V if available
    - get_policy(): Return π if available
    """
    
    def __init__(self, env, gamma: float = 0.99, **kwargs):
        """
        Initialize the agent.
        
        Args:
            env: Environment to interact with
            gamma: Discount factor (how much to value future rewards)
            **kwargs: Additional hyperparameters (stored but not used here)
        """
        self.env = env
        self.gamma = gamma
        self.hyperparams = kwargs
        
        # These will be set by subclasses after training:
        self.V: Optional[np.ndarray] = None    # Value function V(s)
        self.Q: Optional[np.ndarray] = None    # Q-values Q(s,a)
        self.pi: Optional[Dict[int, Any]] = None  # Policy π(s)
        
        # Training status
        self.is_trained = False
        self.training_info: Dict[str, Any] = {}
    
    # =========================================================================
    # ABSTRACT METHODS - Subclasses MUST implement these
    # =========================================================================
    
    @abstractmethod
    def select_action(self, state: int) -> Any:
        """
        Choose an action for the given state.
        
        This is the core decision-making method!
        
        Args:
            state: Current state index
            
        Returns:
            action: The chosen action
        """
        pass  # Subclasses must implement this
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Train the agent (learn policy/value function).
        
        Returns:
            info: Dictionary with training results
        """
        pass  # Subclasses must implement this
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save agent to file."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str, env) -> "BaseAgent":
        """Load agent from file."""
        pass
    
    # =========================================================================
    # PROVIDED METHODS - Already implemented for all agents
    # =========================================================================
    
    def get_value_function(self) -> Optional[np.ndarray]:
        """
        Get the value function V(s) if available.
        
        Returns:
            V: Array where V[s] = value of state s, or None if not trained
        """
        return self.V
    
    def get_q_values(self) -> Optional[np.ndarray]:
        """Get Q-values Q(s,a) if available."""
        return self.Q
    
    def get_policy(self) -> Optional[Dict[int, Any]]:
        """
        Get the policy π(s) if available.
        
        Returns:
            pi: Dictionary where pi[s] = action to take in state s
        """
        return self.pi
    
    def evaluate_episode(self, max_steps: int = 1000) -> Tuple[float, int, str]:
        """
        Run one episode using the learned policy.
        
        HOW IT WORKS:
        -------------
        1. Reset environment to start state
        2. Loop:
           a. Agent selects action using policy
           b. Environment returns next state and reward
           c. If episode ended, stop
        3. Return total reward, steps taken, and outcome
        
        Args:
            max_steps: Safety limit (stop if episode runs too long)
            
        Returns:
            total_reward: Sum of all rewards in the episode
            steps: How many steps were taken
            outcome: "goal", "crash", or "timeout"
        """
        # Start episode
        state, _ = self.env.reset()
        total_reward = 0.0
        
        # Run until episode ends or timeout
        for step in range(max_steps):
            # Agent chooses action
            action = self.select_action(state)
            
            # Environment responds
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Accumulate reward
            total_reward += reward
            
            # Update state
            state = next_state
            
            # Check if episode ended
            if terminated:
                # Episode ended naturally (goal or crash)
                outcome = info.get("terminal", "goal")
                return total_reward, step + 1, outcome
            
            if truncated:
                # Environment truncated the episode
                return total_reward, step + 1, "timeout"
        
        # Hit max_steps without ending
        return total_reward, max_steps, "timeout"
