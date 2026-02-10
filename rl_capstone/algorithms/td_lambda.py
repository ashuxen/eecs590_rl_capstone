"""
Temporal Difference Learning: TD(λ)
===================================

Requirement 4: "For environments without sufficiently simple subtasks, you can trade
this requirement out for backward view TD(lambda) + greedy improvements on expected values."

TD methods learn from experience without requiring a model:
- TD(0): V(s) <- V(s) + α [r + γV(s') - V(s)]
- TD(λ): Uses eligibility traces to blend TD and Monte Carlo

TD(λ) parameter:
- λ=0: Pure TD(0), bootstraps from next state
- λ=1: Monte Carlo, uses full returns
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


def td_zero(env,
            policy: Callable[[int], int],
            gamma: float = 0.99,
            alpha: float = 0.1,
            n_episodes: int = 1000,
            V_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[float]]:
    """
    TD(0) Learning for policy evaluation.
    
    Updates: V(s) <- V(s) + α [r + γV(s') - V(s)]
    
    Args:
        env: Environment with reset() and step(action) methods
        policy: Function mapping state -> action
        gamma: Discount factor
        alpha: Learning rate
        n_episodes: Number of episodes to run
        V_init: Initial value function (zeros if None)
        
    Returns:
        V: Learned value function
        returns: Episode returns for plotting
    """
    n_states = env.n_states
    V = V_init.copy() if V_init is not None else np.zeros(n_states)
    episode_returns = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_return = 0
        
        done = False
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TD(0) update
            td_target = reward + gamma * V[next_state] * (1 - done)
            td_error = td_target - V[state]
            V[state] += alpha * td_error
            
            total_return += reward
            state = next_state
        
        episode_returns.append(total_return)
    
    return V, episode_returns


def td_lambda(env,
              policy: Callable[[int], int],
              gamma: float = 0.99,
              alpha: float = 0.1,
              lam: float = 0.8,
              n_episodes: int = 1000,
              V_init: Optional[np.ndarray] = None,
              replacing_traces: bool = True) -> Tuple[np.ndarray, List[float]]:
    """
    TD(λ) Learning with eligibility traces (backward view).
    
    Uses eligibility traces to credit past states:
    - e(s) <- γλe(s) for all s
    - e(s_t) <- e(s_t) + 1 (or = 1 for replacing traces)
    - V(s) <- V(s) + α δ e(s) for all s
    
    Args:
        env: Environment
        policy: Policy function
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter (0=TD(0), 1=MC)
        n_episodes: Number of episodes
        V_init: Initial value function
        replacing_traces: Use replacing (True) or accumulating (False) traces
        
    Returns:
        V: Learned value function
        returns: Episode returns
    """
    n_states = env.n_states
    V = V_init.copy() if V_init is not None else np.zeros(n_states)
    episode_returns = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_return = 0
        
        # Initialize eligibility traces
        e = np.zeros(n_states)
        
        done = False
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # TD error
            td_target = reward + gamma * V[next_state] * (1 - done)
            td_error = td_target - V[state]
            
            # Update eligibility trace for current state
            if replacing_traces:
                e[state] = 1.0  # Replacing traces
            else:
                e[state] += 1.0  # Accumulating traces
            
            # Update all states using eligibility traces
            V += alpha * td_error * e
            
            # Decay traces
            e *= gamma * lam
            
            total_return += reward
            state = next_state
        
        episode_returns.append(total_return)
    
    return V, episode_returns


def td_lambda_control(env,
                      gamma: float = 0.99,
                      alpha: float = 0.1,
                      lam: float = 0.8,
                      epsilon: float = 0.1,
                      n_episodes: int = 1000) -> Tuple[np.ndarray, Dict[int, int], List[float]]:
    """
    SARSA(λ) - TD(λ) for control (learning Q-values).
    
    Learns Q-values with eligibility traces and ε-greedy policy.
    
    Args:
        env: Environment
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter
        epsilon: Exploration probability
        n_episodes: Number of episodes
        
    Returns:
        Q: Q-values (n_states, n_actions)
        pi: Learned policy
        returns: Episode returns
    """
    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))
    episode_returns = []
    
    def epsilon_greedy(state: int) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(Q[state, :]))
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state)
        total_return = 0
        
        # Eligibility traces for (s, a) pairs
        e = np.zeros((n_states, n_actions))
        
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_action = epsilon_greedy(next_state) if not done else 0
            
            # TD error
            td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
            td_error = td_target - Q[state, action]
            
            # Update trace
            e[state, action] = 1.0  # Replacing trace
            
            # Update Q-values
            Q += alpha * td_error * e
            
            # Decay traces
            e *= gamma * lam
            
            total_return += reward
            state, action = next_state, next_action
        
        episode_returns.append(total_return)
    
    # Extract greedy policy
    pi = {s: int(np.argmax(Q[s, :])) for s in range(n_states)}
    
    return Q, pi, episode_returns


def compute_lambda_return(rewards: List[float],
                          values: np.ndarray,
                          gamma: float,
                          lam: float,
                          states: List[int]) -> float:
    """
    Compute λ-return G_t^λ for a trajectory.
    
    G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t^{(n)}
    
    where G_t^{(n)} is the n-step return.
    """
    T = len(rewards)
    if T == 0:
        return 0.0
    
    # Compute n-step returns for all n
    n_step_returns = []
    for n in range(1, T + 1):
        # G_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
        G_n = sum(gamma**i * rewards[i] for i in range(min(n, T)))
        if n < T:
            G_n += gamma**n * values[states[n]]
        n_step_returns.append(G_n)
    
    # λ-return is weighted average
    G_lambda = 0.0
    for n, G_n in enumerate(n_step_returns):
        weight = (1 - lam) * (lam ** n) if n < len(n_step_returns) - 1 else lam ** n
        G_lambda += weight * G_n
    
    return G_lambda
