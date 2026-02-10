"""
============================================================================
LOAD AND USE A TRAINED AGENT
============================================================================

This script demonstrates how to:
1. Load a previously trained agent from disk
2. Use it to make decisions in the environment
3. Evaluate its performance

Author: Ashutosh Kumar

Run: python load_and_use_agent.py
============================================================================
"""

import os
import sys
import numpy as np

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rl_capstone.environments import WindyChasmEnv
from rl_capstone.agents import DPAgent


def main():
    # =========================================================================
    # CONFIGURATION - Which agent to load?
    # =========================================================================
    
    # Available trained models:
    #   windy_chasm_B0.3.pkl   - 65% success (easy wind)
    #   windy_chasm_B0.35.pkl  - 54% success (medium wind)
    
    model_name = "windy_chasm_B0.3.pkl"
    B = 0.3  # Must match the B value the agent was trained with!
    
    model_path = os.path.join(project_root, "models", "policy_kernel", model_name)
    
    print("=" * 60)
    print("LOADING TRAINED AGENT")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Path:  {model_path}")
    print()
    
    # =========================================================================
    # STEP 1: Create environment (must match training configuration)
    # =========================================================================
    
    env = WindyChasmEnv(B=B, R_goal=20.0, r_crash=5.0, gamma=0.99)
    print(f"Environment: WindyChasm with B={B}")
    print(f"  States:  {env.n_states}")
    print(f"  Actions: {env.actions}")
    print()
    
    # =========================================================================
    # STEP 2: Load the trained agent
    # =========================================================================
    
    agent = DPAgent.load(model_path, env)
    
    print("Agent loaded successfully!")
    print(f"  Method used:    {agent.training_info.get('method', 'unknown')}")
    print(f"  V*(start):      {agent.V[env.start_state_idx]:.2f}")
    print(f"  Is trained:     {agent.is_trained}")
    print()
    
    # =========================================================================
    # STEP 3: Use the agent - Interactive demo
    # =========================================================================
    
    print("-" * 60)
    print("INTERACTIVE DEMO: Watch the agent navigate!")
    print("-" * 60)
    
    # Run a few episodes with detailed output
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        state, _ = env.reset()
        total_reward = 0
        step = 0
        
        while step < 50:  # Safety limit
            # Get current position
            pos = env.mdp.states[state]
            
            # Agent selects action based on learned policy
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print step details
            print(f"  Step {step:2d}: pos={pos}, action={action}, "
                  f"reward={reward:.1f}, next={env.mdp.states[next_state]}")
            
            state = next_state
            
            if done:
                result = "SUCCESS!" if info["terminal"] == "goal" else "CRASHED"
                print(f"  >>> {result} Total reward: {total_reward:.1f}")
                break
    
    # =========================================================================
    # STEP 4: Batch evaluation
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("BATCH EVALUATION (100 episodes)")
    print("-" * 60)
    
    successes = 0
    total_rewards = []
    
    for ep in range(100):
        state, _ = env.reset()
        ep_reward = 0
        
        for _ in range(200):
            action = agent.select_action(state)
            state, reward, done, _, info = env.step(action)
            ep_reward += reward
            
            if done:
                if info["terminal"] == "goal":
                    successes += 1
                break
        
        total_rewards.append(ep_reward)
    
    print(f"  Success rate: {successes}%")
    print(f"  Mean reward:  {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    # =========================================================================
    # STEP 5: Access policy and value function directly
    # =========================================================================
    
    print("\n" + "-" * 60)
    print("ACCESSING POLICY AND VALUE FUNCTION")
    print("-" * 60)
    
    # The policy is stored as a dictionary: pi[state] = best_action
    print("\nOptimal actions for first 5 states:")
    for s in range(5):
        pos = env.mdp.states[s]
        action = agent.pi[s]
        value = agent.V[s]
        print(f"  State {s} {pos}: action={action}, V*={value:.2f}")
    
    # Show policy at start
    start_idx = env.start_state_idx
    print(f"\nAt START (0,3), index {start_idx}:")
    print(f"  Optimal action: {agent.pi[start_idx]}")
    print(f"  Value V*:       {agent.V[start_idx]:.2f}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
