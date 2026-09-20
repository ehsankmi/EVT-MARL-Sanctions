"""
agents.py
---------
FirmAgent class implementing tabular Q-learning for the EVT-MARL framework.

Each agent represents a firm making discrete risk-exposure decisions
(Conservative / Moderate / Aggressive) under political-economic uncertainty.
The reward function can be either risk-neutral (baseline) or CVaR-penalized
(risk-aware), as described in the working paper.
"""

import numpy as np


# Action profiles: (base_return, shock_beta)
# shock_beta determines how much of the realized shock enters the agent's return
ACTION_PROFILES = {
    0: {"name": "Conservative", "base_return": 0.0005, "shock_beta": 0.8},
    1: {"name": "Moderate",     "base_return": 0.0014, "shock_beta": 2.2},
    2: {"name": "Aggressive",   "base_return": 0.0026, "shock_beta": 4.5},
}

N_ACTIONS = len(ACTION_PROFILES)


class FirmAgent:
    """
    A single firm agent using tabular Q-learning.

    State: (cash_ratio_bucket, last_shock_bucket) — 5 × 3 = 15 states
    Actions: 0=Conservative, 1=Moderate, 2=Aggressive
    """

    N_CASH_BUCKETS = 5
    N_SHOCK_BUCKETS = 3
    N_STATES = N_CASH_BUCKETS * N_SHOCK_BUCKETS
    CASH_EDGES = [0.2, 0.6, 1.0, 1.6]

    def __init__(
        self,
        agent_id: int,
        rng: np.random.Generator,
        alpha_lr: float = 0.1,
        gamma: float = 0.95,
    ):
        self.id = agent_id
        self.rng = rng
        self.alpha_lr = alpha_lr
        self.gamma = gamma

        self.Q = np.zeros((self.N_STATES, N_ACTIONS))
        self.cash_ratio = 1.0
        self.alive = True

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    @classmethod
    def cash_bucket(cls, cash_ratio: float) -> int:
        b = 0
        for edge in cls.CASH_EDGES:
            if cash_ratio > edge:
                b += 1
        return min(b, cls.N_CASH_BUCKETS - 1)

    @staticmethod
    def shock_bucket(shock: float, pool_std: float) -> int:
        a = abs(shock)
        if a < pool_std:
            return 0
        elif a < 4 * pool_std:
            return 1
        return 2

    @classmethod
    def state_index(cls, cb: int, sb: int) -> int:
        return cb * cls.N_SHOCK_BUCKETS + sb

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: int, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next = np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha_lr * (
            reward + self.gamma * best_next - self.Q[state, action]
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def dominant_strategy(self) -> str:
        """Return the action name that dominates across all states."""
        counts = np.bincount(
            [np.argmax(self.Q[s]) for s in range(self.N_STATES)],
            minlength=N_ACTIONS,
        )
        return ACTION_PROFILES[int(np.argmax(counts))]["name"]
