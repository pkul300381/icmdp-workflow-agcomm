"""
mocks.py
--------
Test doubles for the ICMDP test suite.

MockIntentParser
    Replaces LLMIntentParser.  Tests can inject an exact list of Predicates
    so that component tests (feasibility, solver, execution engine) are fully
    isolated from the regex / LLM parsing logic.

MockLLMBackend
    Simulates a future LLM backend.  Returns pre-configured structured
    constraint objects so that intent-parsing tests can verify the downstream
    handling of LLM output without making network calls.

DeterministicEnvironment
    Wraps MarketEnvironment and lets tests pin the true state for a vendor,
    ensuring step() returns a known next state regardless of randomness.
"""
from typing import Callable, Dict, List, Optional, Tuple

from environment import Action, MarketEnvironment, State
from intent_parser import LLMIntentParser, Predicate


# ---------------------------------------------------------------------------
# MockIntentParser
# ---------------------------------------------------------------------------

class MockIntentParser:
    """
    Drop-in replacement for LLMIntentParser that returns a caller-supplied
    predicate list regardless of the instruction string.

    Usage
    -----
        mock = MockIntentParser()
        mock.set_predicates([eco_pred, inventory_pred])
        admissible = mock.get_admissible_actions(state, all_actions, mock.parse_intent("anything"))
    """

    def __init__(self, predicates: Optional[List[Predicate]] = None) -> None:
        self._predicates: List[Predicate] = list(predicates or [])

    def set_predicates(self, predicates: List[Predicate]) -> None:
        self._predicates = list(predicates)

    def parse_intent(self, instruction: str) -> List[Predicate]:  # noqa: ARG002
        return list(self._predicates)

    def get_admissible_actions(
        self,
        state: State,
        all_actions: List[Action],
        predicates: List[Predicate],
    ) -> List[Action]:
        return [
            a for a in all_actions
            if a.vendor == state.vendor and all(p(state, a) for p in predicates)
        ]


# ---------------------------------------------------------------------------
# MockLLMBackend
# ---------------------------------------------------------------------------

class MockLLMBackend:
    """
    Simulates a structured-output LLM response for intent parsing tests.

    The backend stores a mapping of instruction → list of constraint dicts
    (mirroring what a real LLM would return as JSON).  Tests can register
    responses with `register()` and verify that downstream code handles the
    structured output correctly.

    Example structured response format
    -----------------------------------
        {
            "constraints": [
                {"type": "eco_constraint",    "hardness": "soft", "priority": 1},
                {"type": "budget_constraint", "hardness": "soft", "priority": 2,
                 "threshold": 200.0},
            ]
        }
    """

    def __init__(self) -> None:
        self._responses: Dict[str, dict] = {}
        self._call_log: List[str] = []

    def register(self, instruction: str, response: dict) -> None:
        """Register a canned LLM response for *instruction*."""
        self._responses[instruction.lower()] = response

    def call(self, instruction: str) -> dict:
        """Return the registered response (or empty constraints if unknown)."""
        self._call_log.append(instruction)
        return self._responses.get(instruction.lower(), {"constraints": []})

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    @property
    def last_instruction(self) -> Optional[str]:
        return self._call_log[-1] if self._call_log else None


# ---------------------------------------------------------------------------
# DeterministicEnvironment
# ---------------------------------------------------------------------------

class DeterministicEnvironment:
    """
    Thin wrapper around MarketEnvironment that lets tests pin the true hidden
    state for a vendor so that env.step() behaves predictably.

    Usage
    -----
        det_env = DeterministicEnvironment(env)
        det_env.set_state("VendorA", State(p_ask=100.0, inv=1, eco=1,
                                           rep=0.9, vendor="VendorA"))
        state, reward, done = det_env.step("VendorA", action)
    """

    def __init__(self, env: MarketEnvironment) -> None:
        self._env = env

    def set_state(self, vendor: str, state: State) -> None:
        self._env.true_states[vendor] = state

    def step(
        self, vendor: str, action: Action, t_elapsed: float = 0.0
    ) -> Tuple[State, float, bool]:
        return self._env.step(vendor, action, t_elapsed)

    def get_all_actions(self) -> List[Action]:
        return self._env.get_all_actions()


# ---------------------------------------------------------------------------
# Predicate factory helpers — build one-off Predicate objects in tests
# ---------------------------------------------------------------------------

def always_true_predicate(name: str = "always_true", hardness: str = "soft",
                           priority: int = 1) -> Predicate:
    """A predicate that always returns True (never filters any action)."""
    return Predicate(name=name, fn=lambda s, a: True, hardness=hardness,
                     relaxation_priority=priority)


def always_false_predicate(name: str = "always_false", hardness: str = "soft",
                            priority: int = 1) -> Predicate:
    """A predicate that always returns False (filters every action)."""
    return Predicate(name=name, fn=lambda s, a: False, hardness=hardness,
                     relaxation_priority=priority)


def eco_predicate(hardness: str = "soft", priority: int = 1) -> Predicate:
    """Eco constraint: s.eco == 1."""
    return Predicate(
        name="eco_constraint",
        fn=lambda s, a: s.eco == 1,
        hardness=hardness,
        relaxation_priority=priority,
    )


def budget_predicate(limit: float, hardness: str = "soft",
                     priority: int = 2) -> Predicate:
    """Budget constraint: bid/pay price <= limit."""
    def fn(s: State, a: Action) -> bool:
        if a.action_type in ("Bid", "CounterBid", "Pay"):
            return a.price <= limit
        if a.action_type == "Accept":
            return s.p_ask <= limit
        return True
    return Predicate(
        name=f"budget_constraint_<={limit}",
        fn=fn,
        hardness=hardness,
        relaxation_priority=priority,
    )


def inventory_predicate() -> Predicate:
    """Hard inventory constraint: inv == 1 for consumptive actions."""
    def fn(s: State, a: Action) -> bool:
        return s.inv == 1 if a.action_type in {"Accept", "Pay", "Bid", "CounterBid"} else True
    return Predicate(
        name="inventory_constraint",
        fn=fn,
        hardness="hard",
        relaxation_priority=0,
    )


def latency_predicate(max_secs: float = 60.0, hardness: str = "soft",
                      priority: int = 3) -> Predicate:
    """Latency constraint: s.t_elapsed <= max_secs."""
    return Predicate(
        name=f"latency_constraint_<={max_secs}s",
        fn=lambda s, a, ms=max_secs: s.t_elapsed <= ms,
        hardness=hardness,
        relaxation_priority=priority,
    )
