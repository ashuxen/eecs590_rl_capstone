"""
============================================================================
TRAIN AND SAVE SCRIPT - V1 Requirement 5 Compliance
============================================================================

This script trains the agent and saves the policy kernel + value function.

Author: Ashutosh Kumar

V1 REQUIREMENT 5 SAYS:
"Also have a copy of your best policy kernel + value function pair
if you've trained it at least once."

THIS SCRIPT:
1. Creates the Windy Chasm environment
2. Trains with Value Iteration (finds V* and π*)
3. Evaluates the trained agent
4. Saves the policy kernel to models/policy_kernel/
5. Generates a results summary

HOW TO RUN:
-----------
Option 1 - With Isaac Sim Python:
    C:\\isaacsim\\IsaacLab\\_isaac_sim\\python.bat train_and_save.py

Option 2 - With regular Python (if scipy/numpy installed):
    python train_and_save.py

OUTPUT:
-------
- models/policy_kernel/windy_chasm_B0.5.pkl  (trained agent)
- models/policy_kernel/windy_chasm_B0.5_history.json  (training history)
- reports/figures/value_function_heatmap.png  (optional, if matplotlib works)

============================================================================
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add the project root to path so we can import our modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    """
    Main training function.
    
    This is the script you run to train and save the agent.
    """
    
    print("="*70)
    print("EECS 590 RL CAPSTONE - V1 Training Script")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # =========================================================================
    # CONFIGURATION - Change these parameters to experiment!
    # =========================================================================
    
    # Environment parameters
    B = 0.3           # Wind probability (0.3=easy, 0.5=medium, 0.7=hard)
    gamma = 0.99      # Discount factor
    R_goal = 20.0     # Reward for reaching goal
    r_crash = 5.0     # Penalty for crashing
    
    # Training parameters
    method = "value_iteration"  # Options: value_iteration, policy_iteration, q_value_iteration
    
    # Evaluation parameters
    num_eval_episodes = 100
    
    # Output paths
    model_dir = os.path.join(project_root, "models", "policy_kernel")
    figures_dir = os.path.join(project_root, "reports", "figures")
    
    print("CONFIGURATION:")
    print(f"  B (wind strength) = {B}")
    print(f"  gamma (discount)  = {gamma}")
    print(f"  Method           = {method}")
    print(f"  Eval episodes    = {num_eval_episodes}")
    print()
    
    # =========================================================================
    # STEP 1: CREATE ENVIRONMENT
    # =========================================================================
    
    print("-"*70)
    print("STEP 1: Creating Windy Chasm Environment")
    print("-"*70)
    
    from rl_capstone.environments import WindyChasmEnv
    
    env = WindyChasmEnv(B=B, R_goal=R_goal, r_crash=r_crash, gamma=gamma)
    
    print(f"  State space:  {env.n_states} states")
    print(f"  Action space: {env.n_actions} actions ({env.actions})")
    print(f"  Start state:  (0, 3) -> index {env.start_state_idx}")
    print()
    
    # =========================================================================
    # STEP 2: CREATE AGENT
    # =========================================================================
    
    print("-"*70)
    print("STEP 2: Creating DP Agent")
    print("-"*70)
    
    from rl_capstone.agents import DPAgent
    
    agent = DPAgent(
        env=env,
        gamma=gamma,
        method=method,
        tol=1e-10,
        max_iter=50000
    )
    
    print(f"  Agent type:   {type(agent).__name__}")
    print(f"  Method:       {agent.method}")
    print(f"  Gamma:        {agent.gamma}")
    print()
    
    # =========================================================================
    # STEP 3: TRAIN (This is where the magic happens!)
    # =========================================================================
    
    print("-"*70)
    print("STEP 3: Training Agent (Solving MDP)")
    print("-"*70)
    
    train_info = agent.train()
    
    print()
    print("TRAINING RESULTS:")
    print(f"  Iterations to converge: {train_info['iterations']}")
    print(f"  V*(start) = V*(0,3)   : {train_info['v_start']:.4f}")
    print(f"  V* mean               : {train_info['v_mean']:.4f}")
    print(f"  V* range              : [{train_info['v_min']:.2f}, {train_info['v_max']:.2f}]")
    print()
    
    # =========================================================================
    # STEP 4: EVALUATE (Test the trained policy)
    # =========================================================================
    
    print("-"*70)
    print(f"STEP 4: Evaluating Agent ({num_eval_episodes} episodes)")
    print("-"*70)
    
    # Run evaluation episodes
    successes = 0
    crashes = 0
    total_rewards = []
    total_steps = []
    
    for ep in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 200:  # Safety limit
            action = agent.select_action(state)
            state, reward, done, _, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                if info.get("terminal") == "goal":
                    successes += 1
                else:
                    crashes += 1
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        # Print progress every 25 episodes
        if (ep + 1) % 25 == 0:
            print(f"  Episode {ep+1}/{num_eval_episodes}: "
                  f"success_rate={successes/(ep+1):.1%}")
    
    success_rate = successes / num_eval_episodes
    crash_rate = crashes / num_eval_episodes
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_steps = np.mean(total_steps)
    
    print()
    print("EVALUATION RESULTS:")
    print(f"  Success rate: {success_rate:.1%} ({successes}/{num_eval_episodes})")
    print(f"  Crash rate:   {crash_rate:.1%} ({crashes}/{num_eval_episodes})")
    print(f"  Mean reward:  {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Mean steps:   {mean_steps:.1f}")
    print()
    
    # =========================================================================
    # STEP 5: SAVE POLICY KERNEL (V1 Requirement 5!)
    # =========================================================================
    
    print("-"*70)
    print("STEP 5: Saving Policy Kernel + Value Function")
    print("-"*70)
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save path
    model_filename = f"windy_chasm_B{B}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save using agent's save method
    agent.save(model_path)
    
    # Also save a summary JSON for easy reference
    summary = {
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "name": "WindyChasm",
            "B": B,
            "gamma": gamma,
            "R_goal": R_goal,
            "r_crash": r_crash,
            "n_states": env.n_states,
            "n_actions": env.n_actions
        },
        "training": {
            "method": method,
            "iterations": train_info["iterations"],
            "v_start": train_info["v_start"],
            "v_mean": train_info["v_mean"]
        },
        "evaluation": {
            "num_episodes": num_eval_episodes,
            "success_rate": success_rate,
            "crash_rate": crash_rate,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_steps": mean_steps
        },
        "files": {
            "model": model_path
        }
    }
    
    summary_path = os.path.join(model_dir, f"windy_chasm_B{B}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Model saved to:   {model_path}")
    print(f"  Summary saved to: {summary_path}")
    print()
    
    # =========================================================================
    # STEP 6: DISPLAY VALUE FUNCTION (Optional visualization)
    # =========================================================================
    
    print("-"*70)
    print("STEP 6: Value Function Summary")
    print("-"*70)
    
    # Get value function as grid
    V = agent.V
    state_to_idx = env.state_to_idx
    
    print("\nV*(i,j) at key positions:")
    print(f"  V*(0,3) = {V[state_to_idx[(0,3)]]:.2f}  <- START")
    print(f"  V*(5,3) = {V[state_to_idx[(5,3)]]:.2f}")
    print(f"  V*(10,3) = {V[state_to_idx[(10,3)]]:.2f}")
    print(f"  V*(15,3) = {V[state_to_idx[(15,3)]]:.2f}")
    print(f"  V*(18,3) = {V[state_to_idx[(18,3)]]:.2f}  <- NEAR GOAL")
    
    # Print simple value grid
    print("\nValue Function Grid (simplified):")
    print("    i:   0      5      10     15     18")
    for j in [6, 5, 4, 3, 2, 1, 0]:
        row = f"j={j}: "
        for i in [0, 5, 10, 15, 18]:
            if (i, j) in state_to_idx:
                v = V[state_to_idx[(i, j)]]
                row += f"{v:6.1f} "
            else:
                row += "   -   "
        if j == 3:
            row += " <- CENTER (start)"
        elif j == 0 or j == 6:
            row += " <- WALL"
        print(row)
    
    # =========================================================================
    # DONE!
    # =========================================================================
    
    print()
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print()
    print("V1 REQUIREMENT 5 SATISFIED:")
    print(f"  [x] Agent trained with {method}")
    print(f"  [x] Policy kernel saved to: {model_path}")
    print(f"  [x] Value function V* computed and saved")
    print(f"  [x] Evaluation completed: {success_rate:.1%} success rate")
    print()
    print("TO LOAD THIS AGENT LATER:")
    print("  from rl_capstone.agents import DPAgent")
    print("  from rl_capstone.environments import WindyChasmEnv")
    print("  env = WindyChasmEnv(B=0.5)")
    print(f"  agent = DPAgent.load('{model_path}', env)")
    print()
    
    return summary


# =============================================================================
# ADDITIONAL EXPERIMENTS (uncomment to run)
# =============================================================================

def run_multiple_B_values():
    """
    Train agents with different wind strengths for comparison.
    
    This creates multiple policy kernels for different B values.
    """
    from rl_capstone.environments import WindyChasmEnv
    from rl_capstone.agents import DPAgent
    
    B_values = [0.3, 0.5, 0.7]
    results = {}
    
    for B in B_values:
        print(f"\n{'='*50}")
        print(f"Training with B = {B}")
        print(f"{'='*50}")
        
        env = WindyChasmEnv(B=B, gamma=0.99)
        agent = DPAgent(env, method="value_iteration")
        info = agent.train()
        
        # Quick evaluation
        successes = 0
        for _ in range(50):
            state, _ = env.reset()
            for _ in range(100):
                action = agent.select_action(state)
                state, _, done, _, info_step = env.step(action)
                if done:
                    if info_step.get("terminal") == "goal":
                        successes += 1
                    break
        
        results[B] = {
            "v_start": info["v_start"],
            "iterations": info["iterations"],
            "success_rate": successes / 50
        }
        
        # Save
        model_dir = os.path.join(project_root, "models", "policy_kernel")
        os.makedirs(model_dir, exist_ok=True)
        agent.save(os.path.join(model_dir, f"windy_chasm_B{B}.pkl"))
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"{'B':>5} | {'V*(start)':>10} | {'Iters':>8} | {'Success':>8}")
    print("-"*50)
    for B, r in results.items():
        print(f"{B:>5.1f} | {r['v_start']:>10.2f} | {r['iterations']:>8} | {r['success_rate']:>7.0%}")
    
    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Run main training
    summary = main()
    
    # Optionally run experiments with different B values
    # Uncomment the line below to train multiple agents:
    # run_multiple_B_values()
