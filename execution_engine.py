"""
execution_engine.py
-------------------
Orchestrates the full Agentic Commerce pipeline (Figure 1 in the paper):

  User Intent → LLM Parse → POMDP Belief Update → ICMDP Feasibility Check
  → Masked Q-Learning → Environment Step → Certificate of Execution

Session-level state (t_elapsed, budget_spent) is tracked here and injected
into the full ICMDP state via make_full_state() before every constraint check.

All training and execution parameters are driven by config.yaml.
"""
import datetime
from typing import Dict, List

from environment import MarketEnvironment, BeliefTracker, State, Action, make_full_state
from intent_parser import LLMIntentParser, Predicate
from feasibility import FeasibilityMonitor
from solvers import MaskedQLearning


class ExecutionEngine:
    """
    The four architectural layers from Table 1 of the paper converge here:

      Layer           Component
      ──────────────  ───────────────────────────────────────────────────
      User / Intent   LLMIntentParser  (f_θ : L × S → 2^A)
      Perception      BeliefTracker    (POMDP belief updater, Eq. 6)
      Decision        MaskedQLearning  (ICMDP solver, Eq. 20)
      Settlement      ExecutionEngine  (certificate of execution, §6.5)
    """

    def __init__(
        self,
        env: MarketEnvironment,
        parser: LLMIntentParser,
        monitor: FeasibilityMonitor,
        agent: MaskedQLearning,
        cfg: dict,
    ) -> None:
        self.env = env
        self.parser = parser
        self.monitor = monitor
        self.agent = agent
        self.tracker = BeliefTracker(env, cfg)
        self.certificate: List[Dict] = []

        sc = cfg["policies"]["solver"]
        self._training_episodes: int = sc["training_episodes"]
        self._max_steps: int = sc["max_steps_per_episode"]

        ss = cfg["state_space"]["session_state"]
        self._step_dur: float = ss.get("step_duration_seconds", 1.0)
        self._initial_budget: float = ss.get("initial_budget", 200.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _full_state(
        self, vendor: str, t_elapsed: float, budget_spent: float, step: int
    ) -> State:
        """
        Attach session-level fields to the current belief state, producing
        the full ICMDP state S = vendor_state × session_state used by the
        feasibility monitor and intent predicates.
        """
        return make_full_state(
            vendor_state=self.tracker.get_most_likely_state(vendor),
            t_elapsed=t_elapsed,
            budget_spent=budget_spent,
            negotiation_round=step,
        )

    # ------------------------------------------------------------------
    # Phase 1: Online training
    # ------------------------------------------------------------------

    def train_online(
        self,
        instruction: str,
        vendor: str,
        episodes: int = None,
        max_steps: int = None,
    ) -> None:
        """
        Online masked Q-learning training loop (paper §4.6.3).

        Parses intent once, then runs `episodes` episodes of up to
        `max_steps` steps each.  Epsilon decays after every episode.
        """
        episodes = episodes or self._training_episodes
        max_steps = max_steps or self._max_steps

        predicates = self.parser.parse_intent(instruction)
        print(f"--- Online Training: '{instruction}' | {episodes} episodes ---")
        print(f"    Constraints: {[p.name for p in predicates]}")

        for ep in range(episodes):
            raw_state = self.env.reset(vendor)
            self.tracker.estimated_states[vendor] = raw_state

            # Session state — reset each episode
            t_elapsed: float = 0.0
            budget_spent: float = 0.0

            for step in range(max_steps):
                b_state = self._full_state(vendor, t_elapsed, budget_spent, step)

                A_I, feasible, active_preds = self.monitor.check_and_relax(
                    b_state, self.env.get_all_actions(), predicates, self.parser
                )
                if not feasible:
                    break

                action = self.agent.select_action(b_state, A_I)
                next_raw, reward, done = self.env.step(vendor, action, t_elapsed)

                # Advance session counters
                t_elapsed += self._step_dur
                if action.action_type == "Pay" and reward > 0:
                    budget_spent += action.price

                self.tracker.update_belief(action, next_raw, vendor)
                next_b = self._full_state(vendor, t_elapsed, budget_spent, step + 1)
                next_A_I = self.parser.get_admissible_actions(
                    next_b, self.env.get_all_actions(), active_preds
                )

                # Masked Q-update (Eq. 20)
                self.agent.update(b_state, action, reward, next_b, next_A_I)

                if done:
                    break

            self.agent.decay_epsilon()

        print("--- Training Complete ---")

    # ------------------------------------------------------------------
    # Phase 2: Execution with certificate generation
    # ------------------------------------------------------------------

    def execute_transaction(
        self,
        instruction: str,
        vendor: str,
        max_steps: int = None,
    ) -> List[Dict]:
        """
        Execute a greedy transaction using the trained policy and produce
        a Certificate of Execution (paper §6.5).

        The certificate provides formal evidence that every action was within
        A_I(s_t) at each step t — a guarantee unavailable in reward-shaped
        or expectation-constrained systems.
        """
        max_steps = max_steps or self._max_steps
        print(f"--- Executing Transaction: '{instruction}' ---")
        self.certificate = []

        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0   # greedy execution

        predicates = self.parser.parse_intent(instruction)
        raw_state = self.env.reset(vendor)
        self.tracker.estimated_states[vendor] = raw_state

        t_elapsed: float = 0.0
        budget_spent: float = 0.0

        for step in range(max_steps):
            b_state = self._full_state(vendor, t_elapsed, budget_spent, step)

            A_I, feasible, active_preds = self.monitor.check_and_relax(
                b_state, self.env.get_all_actions(), predicates, self.parser
            )
            if not feasible:
                print("Transaction aborted: A_I(s) = ∅, constraints infeasible.")
                self.certificate.append({
                    "step": step,
                    "timestamp": str(datetime.datetime.now()),
                    "error": "Transaction aborted — A_I(s) = ∅",
                    "belief_state": str(b_state),
                    "t_elapsed_s": t_elapsed,
                    "budget_spent": budget_spent,
                })
                break

            action = self.agent.select_action(b_state, A_I)
            next_raw, reward, done = self.env.step(vendor, action, t_elapsed)

            t_elapsed += self._step_dur
            if action.action_type == "Pay" and reward > 0:
                budget_spent += action.price

            # Certificate entry for this step (paper §6.5: b_t, A_I(s_t), a_t*)
            self.certificate.append({
                "step": step,
                "timestamp": str(datetime.datetime.now()),
                "belief_state": str(b_state),
                "t_elapsed_s": round(t_elapsed, 3),
                "budget_spent": round(budget_spent, 4),
                "admissible_actions": [str(a) for a in A_I],
                "selected_action": str(action),
                "reward": round(reward, 4),
                "constraints_satisfied": True,   # guaranteed by construction (ICMDP)
                "active_constraints": [
                    {"name": p.name, "hardness": p.hardness,
                     "relaxation_priority": p.relaxation_priority}
                    for p in active_preds
                ],
            })

            self.tracker.update_belief(action, next_raw, vendor)
            if done:
                print(f"Transaction complete.  reward={reward:.1f}  "
                      f"t={t_elapsed:.1f}s  spent=${budget_spent:.2f}")
                break

        self.agent.epsilon = original_epsilon
        return self.certificate
