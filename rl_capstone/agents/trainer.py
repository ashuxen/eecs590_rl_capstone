"""
============================================================================
TRAINER - Manages Training, Evaluation, and Saving
============================================================================

WHY THIS FILE IS IMPORTANT:
---------------------------
The Trainer is the "coach" of your RL system. It:
1. Orchestrates training (calls agent.train())
2. Runs evaluation episodes to test performance
3. Saves the best policy for later use
4. Keeps track of results and history

This is your main interface for running experiments!

TYPICAL WORKFLOW:
-----------------
    env = WindyChasmEnv(B=0.5)
    agent = DPAgent(env)
    trainer = Trainer(agent, env)
    
    trainer.train()                    # Learn optimal policy
    trainer.evaluate(num_episodes=100) # Test it
    trainer.save("models/best.pkl")    # Save for later

============================================================================
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent


@dataclass
class EvaluationResult:
    """
    Stores results from evaluating an agent.
    
    This is like a report card for your agent!
    
    Attributes:
        success_rate: Fraction of episodes that reached the goal
        crash_rate: Fraction of episodes that crashed
        timeout_rate: Fraction that ran too long
        mean_reward: Average total reward per episode
        std_reward: Standard deviation of rewards
        mean_steps: Average steps per episode
        std_steps: Standard deviation of steps
    """
    success_rate: float
    crash_rate: float
    timeout_rate: float
    mean_reward: float
    std_reward: float
    mean_steps: float
    std_steps: float
    num_episodes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "success_rate": self.success_rate,
            "crash_rate": self.crash_rate,
            "timeout_rate": self.timeout_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_steps": self.mean_steps,
            "std_steps": self.std_steps,
            "num_episodes": self.num_episodes
        }


class Trainer:
    """
    Manages training and evaluation of RL agents.
    
    WHAT IT DOES:
    -------------
    - train(): Calls agent.train() and records results
    - evaluate(): Runs episodes and computes statistics
    - save(): Saves agent + history to disk
    - load(): Loads previously saved agent
    
    WHY USE A TRAINER?
    ------------------
    - Keeps agent code clean (agent just learns, trainer manages)
    - Handles logging and history
    - Makes experiments reproducible
    - Provides a standard interface
    
    Usage:
        trainer = Trainer(agent, env)
        trainer.train()
        results = trainer.evaluate(100)
        print(f"Success rate: {results.success_rate:.1%}")
        trainer.save("models/my_agent.pkl")
    """
    
    def __init__(self, agent: BaseAgent, env, log_dir: str = "models"):
        """
        Create a trainer.
        
        Args:
            agent: The agent to train (e.g., DPAgent)
            env: The environment (e.g., WindyChasmEnv)
            log_dir: Where to save models
        """
        self.agent = agent
        self.env = env
        self.log_dir = log_dir
        
        # Keep track of training runs and evaluations
        self.training_history: List[Dict[str, Any]] = []
        self.evaluation_history: List[EvaluationResult] = []
        
        # Track best performance
        self.best_v_start: float = -np.inf
        self.best_agent_path: Optional[str] = None
    
    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Train the agent.
        
        This is a wrapper around agent.train() that also:
        - Prints progress
        - Records results
        - Tracks best performance
        
        Returns:
            info: Training results from the agent
        """
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        print(f"Agent: {type(self.agent).__name__}")
        print(f"Method: {self.agent.method}")
        print(f"Gamma: {self.agent.gamma}")
        
        # Call agent's train method
        info = self.agent.train()
        
        # Record this training run
        record = {
            "timestamp": datetime.now().isoformat(),
            **info
        }
        self.training_history.append(record)
        
        # Check if this is the best so far
        v_start = info.get("v_start", -np.inf)
        if v_start > self.best_v_start:
            self.best_v_start = v_start
            print(f"\n*** New best V*(start) = {v_start:.4f} ***")
        
        return info
    
    def evaluate(self, num_episodes: int = 100,
                 max_steps: int = 1000,
                 verbose: bool = True) -> EvaluationResult:
        """
        Evaluate the trained agent by running episodes.
        
        WHAT THIS DOES:
        ---------------
        1. Reset environment
        2. Agent selects actions until episode ends
        3. Record outcome (success/crash) and total reward
        4. Repeat for num_episodes
        5. Compute statistics
        
        Args:
            num_episodes: How many episodes to run
            max_steps: Safety limit per episode
            verbose: Print progress?
            
        Returns:
            result: EvaluationResult with all statistics
        """
        if not self.agent.is_trained:
            raise ValueError("Agent not trained! Call train() first.")
        
        print("\n" + "="*60)
        print(f"EVALUATING ({num_episodes} episodes)")
        print("="*60)
        
        # Track outcomes
        rewards = []
        steps = []
        outcomes = {"goal": 0, "crash": 0, "timeout": 0}
        
        for ep in range(num_episodes):
            # Run one episode
            reward, n_steps, outcome = self.agent.evaluate_episode(max_steps)
            
            rewards.append(reward)
            steps.append(n_steps)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            
            # Print progress every 20 episodes
            if verbose and (ep + 1) % 20 == 0:
                print(f"  Episode {ep + 1}/{num_episodes}: "
                      f"{outcome}, steps={n_steps}, reward={reward:.1f}")
        
        # Compute statistics
        result = EvaluationResult(
            success_rate=outcomes.get("goal", 0) / num_episodes,
            crash_rate=outcomes.get("crash", 0) / num_episodes,
            timeout_rate=outcomes.get("timeout", 0) / num_episodes,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_steps=float(np.mean(steps)),
            std_steps=float(np.std(steps)),
            num_episodes=num_episodes
        )
        
        # Save to history
        self.evaluation_history.append(result)
        
        # Print summary
        if verbose:
            print("\n" + "-"*40)
            print("RESULTS:")
            print("-"*40)
            print(f"  Success rate: {result.success_rate:.1%}")
            print(f"  Crash rate:   {result.crash_rate:.1%}")
            print(f"  Mean reward:  {result.mean_reward:.2f} +/- {result.std_reward:.2f}")
            print(f"  Mean steps:   {result.mean_steps:.1f} +/- {result.std_steps:.1f}")
        
        return result
    
    def save(self, filepath: str) -> str:
        """
        Save the agent and training history.
        
        This saves:
        - The trained agent (policy + value function)
        - Training history
        - Evaluation history
        
        Args:
            filepath: Where to save (e.g., "models/policy_kernel/v1.pkl")
            
        Returns:
            filepath: The actual path saved to
        """
        # Create directory if needed
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save agent
        self.agent.save(filepath)
        self.best_agent_path = filepath
        
        # Save history as JSON
        history_path = filepath.replace(".pkl", "_history.json")
        history = {
            "training": self.training_history,
            "evaluation": [r.to_dict() for r in self.evaluation_history],
            "best_v_start": self.best_v_start
        }
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved history to {history_path}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load a previously saved agent.
        
        Args:
            filepath: Where to load from
        """
        # Load agent
        agent_class = type(self.agent)
        self.agent = agent_class.load(filepath, self.env)
        self.best_agent_path = filepath
        
        # Try to load history
        history_path = filepath.replace(".pkl", "_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            self.training_history = history.get("training", [])
            self.best_v_start = history.get("best_v_start", -np.inf)
            print(f"Loaded history from {history_path}")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Command-line interface for training and evaluation.
    
    Run with:
        python -m rl_capstone.agents.trainer --B 0.5 --method value_iteration
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train/Evaluate RL Agent")
    parser.add_argument("--method", 
                        choices=["value_iteration", "q_value_iteration", "policy_iteration"],
                        default="value_iteration", 
                        help="DP method to use")
    parser.add_argument("--B", type=float, default=0.5, 
                        help="Wind probability (0.3=easy, 0.7=hard)")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Evaluation episodes")
    parser.add_argument("--save", type=str, default=None, 
                        help="Path to save trained agent")
    parser.add_argument("--load", type=str, default=None, 
                        help="Path to load agent")
    parser.add_argument("--evaluate", action="store_true", 
                        help="Only evaluate (requires --load)")
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from ..environments import WindyChasmEnv
    from .dp_agent import DPAgent
    
    print("\n" + "="*60)
    print("EECS 590 RL Capstone - Trainer")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Environment: Windy Chasm")
    print(f"  B (wind):    {args.B}")
    print(f"  gamma:       {args.gamma}")
    print(f"  Method:      {args.method}")
    
    # Create environment
    env = WindyChasmEnv(B=args.B, gamma=args.gamma)
    
    # Create agent
    agent = DPAgent(env, gamma=args.gamma, method=args.method)
    
    # Create trainer
    trainer = Trainer(agent, env)
    
    # Load if specified
    if args.load:
        trainer.load(args.load)
    
    # Train if not just evaluating
    if not args.evaluate:
        trainer.train()
    
    # Always evaluate
    trainer.evaluate(num_episodes=args.episodes)
    
    # Save if specified
    if args.save:
        trainer.save(args.save)


if __name__ == "__main__":
    main()
