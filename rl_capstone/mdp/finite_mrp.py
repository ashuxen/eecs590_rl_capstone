"""
Finite Markov Reward Process (MRP)
==================================

An MRP is defined by (S, P, R, γ):
- S: finite state space
- P: transition probability matrix P(s'|s)
- R: reward vector R(s)
- γ: discount factor

This module provides the foundation for Mini 1.
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List, Any, Tuple, Optional


@dataclass
class FiniteMRP:
    """
    Finite Markov Reward Process.
    
    Attributes:
        states: List of state labels
        P: Transition matrix (n x n), can be sparse
        R: Reward vector (length n)
        gamma: Discount factor in [0, 1]
    """
    states: List[Any]
    P: np.ndarray  # or sp.csr_matrix
    R: np.ndarray
    gamma: float
    
    # Cached values
    _V: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate inputs."""
        n = len(self.states)
        assert self.P.shape == (n, n), f"P shape {self.P.shape} != ({n}, {n})"
        assert len(self.R) == n, f"R length {len(self.R)} != {n}"
        assert 0 <= self.gamma <= 1, f"gamma {self.gamma} not in [0, 1]"
    
    @property
    def n_states(self) -> int:
        """Number of states."""
        return len(self.states)
    
    def bellman_equation(self, V: np.ndarray) -> np.ndarray:
        """
        Apply Bellman equation: V' = R + γPV
        
        Args:
            V: Current value function estimate
            
        Returns:
            V': Updated value function
        """
        if sp.issparse(self.P):
            return self.R + self.gamma * (self.P @ V)
        return self.R + self.gamma * (self.P @ V)
    
    def solve_value_function(self, method: str = "iteration",
                             tol: float = 1e-10, 
                             max_iter: int = 50000) -> Tuple[np.ndarray, dict]:
        """
        Solve for the value function V = (I - γP)^{-1} R.
        
        Args:
            method: "iteration" or "linear" (direct solve)
            tol: Convergence tolerance
            max_iter: Maximum iterations
            
        Returns:
            V: Value function
            info: Dict with convergence info
        """
        if method == "linear":
            return self._solve_linear()
        else:
            return self._solve_iteration(tol, max_iter)
    
    def _solve_iteration(self, tol: float, max_iter: int) -> Tuple[np.ndarray, dict]:
        """Value iteration: V <- R + γPV."""
        V = np.zeros(self.n_states)
        deltas = []
        
        for iteration in range(max_iter):
            V_new = self.bellman_equation(V)
            delta = np.max(np.abs(V_new - V))
            deltas.append(delta)
            V = V_new
            
            if delta < tol:
                break
        
        self._V = V
        return V, {"iterations": iteration + 1, "deltas": np.array(deltas), "converged": delta < tol}
    
    def _solve_linear(self) -> Tuple[np.ndarray, dict]:
        """Direct solve: (I - γP)V = R."""
        n = self.n_states
        
        if sp.issparse(self.P):
            A = sp.eye(n, format='csr') - self.gamma * self.P
            V = sp.linalg.spsolve(A, self.R)
        else:
            A = np.eye(n) - self.gamma * self.P
            V = np.linalg.solve(A, self.R)
        
        self._V = np.asarray(V)
        return self._V, {"iterations": 1, "method": "linear"}
    
    def get_value(self, state_idx: int) -> float:
        """Get value of a state (solve if needed)."""
        if self._V is None:
            self.solve_value_function()
        return float(self._V[state_idx])
