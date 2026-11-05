"""RL controller for the compression search (gradient-bandit / REINFORCE).

After supervised training produces the teacher, the self-improvement loop treats
"which compact architecture to try next" as a reinforcement-learning problem: each
architecture is an action, distilling+evaluating it is the environment, and the
reward is ``macro_f1 - lambda * size/budget``. A softmax preference over the
discrete actions is updated by the gradient-bandit rule (Sutton & Barto §2.8) so
trials concentrate on high-reward (small, accurate) architectures rather than
sweeping the grid blindly.

Kept to a bandit, not a deep policy net: the action set is a handful of discrete
archs, so a preference vector + running-baseline REINFORCE update is the right size.
Fully seeded -> reproducible.
"""

from __future__ import annotations

import numpy as np


class GradientBanditController:
    """Softmax-preference bandit updated by REINFORCE with a running baseline."""

    def __init__(
        self, n_actions: int, seed: int = 0, lr: float = 0.2, temperature: float = 1.0
    ) -> None:
        if n_actions < 1:
            raise ValueError("need at least one action")
        self.prefs = np.zeros(n_actions, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.temperature = temperature
        self._baseline = 0.0
        self._steps = 0

    def probs(self) -> np.ndarray:
        """Current softmax policy over actions."""
        z = self.prefs / self.temperature
        z = z - z.max()  # stable softmax
        e = np.exp(z)
        return e / e.sum()

    def select(self) -> int:
        """Sample an action from the current policy."""
        return int(self.rng.choice(len(self.prefs), p=self.probs()))

    def update(self, action: int, reward: float) -> None:
        """REINFORCE gradient-bandit update: pull preference toward above-baseline reward."""
        self._steps += 1
        self._baseline += (reward - self._baseline) / self._steps  # running mean
        p = self.probs()
        onehot = np.zeros_like(self.prefs)
        onehot[action] = 1.0
        self.prefs += self.lr * (reward - self._baseline) * (onehot - p)

    def best_action(self) -> int:
        """The action with the highest learned preference (greedy exploit)."""
        return int(np.argmax(self.prefs))
