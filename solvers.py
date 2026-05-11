"""
solvers.py
----------
Intent-constrained Q-learning solver (paper §4.6.3, Eq. 20).

The masked Q-learning update restricts max over the *admissible* action set
A_I(s') rather than the full action space A, enforcing the ICMDP Bellman
optimality equation (Eq. 13/15).

All hyperparameters are driven by config.yaml:
  policies.solver.alpha / gamma / epsilon / epsilon_decay / epsilon_min
"""
import random
from typing import Dict, List, Tuple

from environment import Action, State


class MaskedQLearning:
    """
    Model-free masked Q-learning for ICMDP (paper §4.6.3).

    Q-table keys use State.q_key() which covers only vendor-state dimensions
    (p_ask, inv, eco, rep, vendor).  Session dimensions (t_elapsed,
    budget_spent, round) are excluded so that the learned Q-values generalise
    across episodes that share the same vendor situation.
    """

    def __init__(self, cfg: dict) -> None:
        sc = cfg["policies"]["solver"]
        self.alpha: float = sc["alpha"]
        self.gamma: float = sc["gamma"]
        self.epsilon: float = sc["epsilon"]
        self._epsilon_decay: float = sc.get("epsilon_decay", 1.0)
        self._epsilon_min: float = sc.get("epsilon_min", 0.01)
        # Q-table: (state_key, action_str) → Q-value
        self._q: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Q-table accessors — always use q_key() to stay session-invariant
    # ------------------------------------------------------------------

    def get_q(self, state: State, action: Action) -> float:
        return self._q.get((state.q_key(), str(action)), 0.0)

    def set_q(self, state: State, action: Action, value: float) -> None:
        self._q[(state.q_key(), str(action))] = value

    # ------------------------------------------------------------------
    # Epsilon decay — call once per episode after the training loop
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(self._epsilon_min, self.epsilon * self._epsilon_decay)

    # ------------------------------------------------------------------
    # Action selection — ε-greedy over admissible actions only
    # ------------------------------------------------------------------

    def select_action(self, state: State, admissible: List[Action]) -> Action:
        """
        ε-greedy selection restricted to *admissible* (i.e. A_I(s)).
        Inadmissible actions are never considered, enforcing zero-violation
        per-step admissibility (paper Eq. 10).
        """
        if not admissible:
            raise ValueError(
                "select_action called with an empty admissible set. "
                "The feasibility monitor should have caught this."
            )
        if random.random() < self.epsilon:
            return random.choice(admissible)

        # Greedy: argmax Q over admissible actions (break ties randomly)
        best_q: float = float("-inf")
        best: List[Action] = []
        for a in admissible:
            q = self.get_q(state, a)
            if q > best_q:
                best_q, best = q, [a]
            elif q == best_q:
                best.append(a)
        return random.choice(best)

    # ------------------------------------------------------------------
    # Q-learning update — Eq. (20) in the paper
    # ------------------------------------------------------------------

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_admissible: List[Action],
    ) -> None:
        """
        Q_I(s,a) ← Q_I(s,a) + α [r + γ · max_{a'∈A_I(s')} Q_I(s',a') − Q_I(s,a)]

        The max is taken over next_admissible = A_I(s'), not over A,
        which is the defining modification of masked Q-learning for ICMDP.
        """
        old_q = self.get_q(state, action)
        next_max = (
            max(self.get_q(next_state, a) for a in next_admissible)
            if next_admissible
            else 0.0   # terminal or infeasible next state
        )
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.set_q(state, action, new_q)
