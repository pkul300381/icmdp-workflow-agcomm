"""
feasibility.py
--------------
Feasibility monitor for the ICMDP.

Implements Definition 4.2 (intent feasibility) and the constraint-relaxation
meta-decision layer described in §4.3:

  If A_I(s) = ∅ the system must either:
    (a) relax the constraint I(s*)  [constraint softening], or
    (b) surface a Constraint Violation Warning  [escalate / abort].

All behaviour is driven by config.yaml:
  policies.feasibility.max_relaxations
  policies.feasibility.infeasibility_response
  policies.feasibility.relaxation_cooldown_steps
"""
from typing import List, Tuple

from environment import Action, State
from intent_parser import LLMIntentParser, Predicate


class FeasibilityMonitor:
    """
    Checks whether A_I(s) is non-empty and, if not, progressively relaxes
    soft constraints in ascending relaxation_priority order (lowest priority
    number = dropped first).

    Hard constraints (relaxation_priority == 0) are never dropped — this
    enforces the zero-violation guarantee of ICMDP (paper Eq. 10/25).
    """

    def __init__(self, cfg: dict) -> None:
        feas = cfg["policies"]["feasibility"]
        self.max_relaxations: int = feas["max_relaxations"]
        self._response: str = feas.get("infeasibility_response", "relax")
        self._cooldown: int = feas.get("relaxation_cooldown_steps", 0)
        self._cooldown_counter: int = 0

    def check_and_relax(
        self,
        state: State,
        all_actions: List[Action],
        predicates: List[Predicate],
        parser: LLMIntentParser,
    ) -> Tuple[List[Action], bool, List[Predicate]]:
        """
        Return (A_I(s), is_feasible, active_predicates).

        If A_I(s) is non-empty the current predicates are returned unchanged.
        If A_I(s) is empty, the configured infeasibility response is triggered:

          "relax"           — drop soft predicates one at a time (lowest
                              relaxation_priority first) until A_I(s) ≠ ∅
                              or max_relaxations is exhausted.
          "abort"           — return ([], False, predicates) immediately.
          "escalate_human"  — same as abort; caller should issue EscalateToHuman.
          "notify_user"     — same as abort; caller should surface a warning.
        """
        A_I = parser.get_admissible_actions(state, all_actions, predicates)
        if A_I:
            return A_I, True, predicates

        print(f"[FeasibilityMonitor] A_I(s) = ∅  for  {state}")

        if self._response in ("abort", "escalate_human", "notify_user"):
            print(f"[FeasibilityMonitor] Response={self._response}. Returning INFEASIBLE.")
            return [], False, predicates

        # ---- Progressive constraint relaxation ----------------------------
        # Soft predicates are already sorted ascending by relaxation_priority
        # in LLMIntentParser.parse_intent(), so pop(0) drops the softest first.
        soft = [p for p in predicates if p.hardness == "soft"]
        hard = [p for p in predicates if p.hardness == "hard"]

        if not soft:
            print("[FeasibilityMonitor] No soft constraints to relax. INFEASIBLE.")
            return [], False, predicates

        print("[FeasibilityMonitor] Attempting progressive constraint relaxation...")

        relaxed_soft = list(soft)
        relaxations = 0

        while relaxations < self.max_relaxations and relaxed_soft:
            dropped = relaxed_soft.pop(0)   # remove the softest constraint
            print(f"[FeasibilityMonitor]   Relaxed: '{dropped.name}' "
                  f"(priority={dropped.relaxation_priority})")
            relaxations += 1
            A_I = parser.get_admissible_actions(
                state, all_actions, relaxed_soft + hard
            )
            if A_I:
                print(f"[FeasibilityMonitor] Feasible after {relaxations} relaxation(s).")
                return A_I, True, relaxed_soft + hard

        print("[FeasibilityMonitor] Relaxation budget exhausted. INFEASIBLE.")
        return [], False, predicates
