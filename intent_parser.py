"""
intent_parser.py
----------------
LLM-based probabilistic intent parser (paper §5, Definition 5.1).

The LLMIntentParser translates a natural language instruction into a
prioritised list of Predicate objects that define the intent-admissible
action set A_I(s) = {a ∈ A : all predicates(s, a) are True}.

Constraint types, their hardness, relaxation priority, and activation
keywords are all driven by config.yaml — no code change is needed to
add, remove, or reorder constraints.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional
import re

from environment import Action, State
from secret_manager import SecretManager


# =============================================================================
# Predicate — the typed unit of an intent constraint  I(s) ⊆ A
# =============================================================================

@dataclass
class Predicate:
    """
    A named, typed constraint predicate.

    Attributes
    ----------
    name                 Human-readable identifier used in certificates and logs.
    fn                   The boolean test (state, action) → bool.
    hardness             "hard" → zero-violation, never relaxed (paper Eq. 10/25).
                         "soft" → relaxable in infeasibility recovery (§4.3).
    relaxation_priority  0 = never relax (always used for hard constraints).
                         1..N = soft constraints; *lower* number → dropped first
                         when the feasibility monitor runs constraint relaxation.
    """
    name: str
    fn: Callable[[State, Action], bool]
    hardness: str            # "hard" | "soft"
    relaxation_priority: int # 0 = never; 1 = softest (drop first)

    def __call__(self, state: State, action: Action) -> bool:
        return self.fn(state, action)


# =============================================================================
# LLM Intent Parser  f_θ : L × S → 2^A
# =============================================================================

class LLMIntentParser:
    """
    Translates a natural language instruction into a list of Predicates
    that collectively define I(s) for the ICMDP (paper §5, Algorithm 1).

    The returned list is ordered:
      1. Soft predicates, ascending by relaxation_priority (cheapest to drop first).
      2. Hard predicates at the end (never dropped by the feasibility monitor).

    In production, the `backend` config key switches between regex (default),
    an Anthropic/OpenAI LLM call, or a local model.  The constraint *types*,
    *keywords*, and *priorities* are always sourced from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self._defs: dict = cfg["intent_constraints"]["definitions"]
        self._parser_cfg: dict = cfg["intent_constraints"]["parser"]
        self._session_cfg: dict = cfg["state_space"]["session_state"]
        self._secret_manager = SecretManager(cfg)
        self._api_key = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_intent(self, instruction: str) -> List[Predicate]:
        """
        Parse *instruction* and return the ordered list of active Predicates.

        Soft predicates are sorted ascending by relaxation_priority so the
        feasibility monitor can pop from the front to drop the softest
        constraint first.
        """
        backend = self._parser_cfg.get("backend", "regex")
        
        if backend == "llm_anthropic":
            return self._call_anthropic(instruction)
        elif backend == "llm_openai":
            return self._call_openai(instruction)
        else:
            print(f"[LLMIntentParser] Warning: Unknown backend '{backend}'. Falling back to regex.")
            return self._parse_intent_regex(instruction)

    def _parse_intent_regex(self, instruction: str) -> List[Predicate]:
        """Original regex-based parsing logic."""
        instruction_lower = instruction.lower()
        soft: List[Predicate] = []
        hard: List[Predicate] = []

        for cid, defn in self._defs.items():
            pred = self._build_predicate(cid, defn, instruction_lower)
            if pred is None:
                continue
            if pred.hardness == "hard":
                hard.append(pred)
            else:
                soft.append(pred)

        soft.sort(key=lambda p: p.relaxation_priority)
        return soft + hard

    def _parse_intent_llm(self, instruction: str, backend: str) -> List[Predicate]:
        """
        LLM-based parsing (Algorithm 1).
        Fetches secret via SecretManager and calls the configured LLM backend.
        """
        if not self._api_key:
            self._api_key = self._secret_manager.get_secret()
            if not self._api_key:
                print(f"[LLMIntentParser] Error: Could not retrieve API key for {backend}. Falling back to regex.")
                return self._parse_intent_regex(instruction)

        if backend == "llm_anthropic":
            return self._call_anthropic(instruction)
        
        return self._parse_intent_regex(instruction)

    def _call_anthropic(self, instruction: str) -> List[Predicate]:
        """Calls Anthropic Claude API to extract active constraints."""
        try:
            import anthropic
            import json
            client = anthropic.Anthropic(api_key=self._api_key)
            
            constraint_info = "\n".join([
                f"- {cid}: {d.get('description', '')}" 
                for cid, d in self._defs.items()
            ])
            
            system_prompt = f"""
            You are an Agentic Commerce Intent Parser (Definition 5.1).
            Your task is to translate a natural language instruction into a JSON list of active constraint IDs.
            
            Available constraints:
            {constraint_info}
            
            Return ONLY a JSON list of strings representing the IDs of the constraints that apply.
            Include any 'always_active' constraints by default if they are relevant or hard requirements.
            
            Example output: ["eco_constraint", "budget_constraint", "inventory_constraint"]
            """
            
            response = client.messages.create(
                model=self._parser_cfg.get("model", "claude-3-sonnet-20240229"),
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": instruction}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Simple extraction in case LLM adds conversational filler
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                active_ids = json.loads(json_match.group(0))
            else:
                print("[LLMIntentParser] Warning: Could not parse LLM response. Falling back to regex.")
                return self._parse_intent_regex(instruction)
            
            soft: List[Predicate] = []
            hard: List[Predicate] = []
            
            instruction_lower = instruction.lower()
            for cid, defn in self._defs.items():
                # Activate if LLM explicitly named it OR if it's always_active
                is_active = (cid in active_ids) or defn.get("always_active", False)
                
                if is_active:
                    pred = self._build_predicate(cid, defn, instruction_lower, override_active=True)
                    if pred:
                        if pred.hardness == "hard":
                            hard.append(pred)
                        else:
                            soft.append(pred)
            
            soft.sort(key=lambda p: p.relaxation_priority)
            return soft + hard

        except ImportError:
            print("[LLMIntentParser] Error: 'anthropic' library not installed. pip install anthropic")
            return self._parse_intent_regex(instruction)
        except Exception as e:
            print(f"[LLMIntentParser] Error during Anthropic call: {e}")
            return self._parse_intent_regex(instruction)

    def _call_openai(self, instruction: str) -> List[Predicate]:
        """Placeholder for OpenAI backend."""
        print("[LLMIntentParser] OpenAI backend not yet implemented. Falling back to regex.")
        return self._parse_intent_regex(instruction)

    def get_admissible_actions(
        self,
        state: State,
        all_actions: List[Action],
        predicates: List[Predicate],
    ) -> List[Action]:
        """
        Returns A_I(s) = {a ∈ A : vendor matches AND all predicates pass}.

        This implements the state-dependent action filter I(s) ⊆ A
        described in Definition 4.1 and Eq. (11).
        """
        return [
            a for a in all_actions
            if a.vendor == state.vendor and all(p(state, a) for p in predicates)
        ]

    # ------------------------------------------------------------------
    # Private predicate builders — one per constraint type in config
    # ------------------------------------------------------------------

    def _build_predicate(
        self, cid: str, defn: dict, instruction: str, override_active: bool = False
    ) -> Optional[Predicate]:
        """
        Attempt to construct a Predicate for constraint *cid* given the
        lowercased *instruction*.  Returns None if the constraint is not
        triggered by this instruction (and always_active is false).

        If override_active is True, the keyword check is bypassed (used by LLM).
        """
        ctype: str = defn["type"]
        priority: int = defn.get("relaxation_priority", 0)
        always: bool = defn.get("always_active", False)
        keywords: List[str] = defn.get("keywords", [])

        # ---- inventory_constraint ----------------------------------------
        if cid == "inventory_constraint":
            def fn(s: State, a: Action) -> bool:
                return s.inv == 1 if a.action_type in {
                    "Accept", "Pay", "Bid", "CounterBid"
                } else True
            return Predicate("inventory_constraint", fn, "hard", 0)

        # ---- eco_constraint -----------------------------------------------
        if cid == "eco_constraint":
            if not override_active and not always and not any(kw in instruction for kw in keywords):
                return None
            def fn(s: State, a: Action) -> bool:
                return s.eco == 1
            return Predicate("eco_constraint", fn, ctype, priority)

        # ---- budget_constraint --------------------------------------------
        if cid == "budget_constraint":
            match = re.search(
                r'(?:max|under|no more than|spending|budget|spend|cost|price)'
                r'[^\d]*\$?\s*(\d+(?:\.\d+)?)',
                instruction,
            )
            if not override_active and not match:
                return None
            budget = float(match.group(1)) if match else self._session_cfg.get("initial_budget", 200.0)
            def fn(s: State, a: Action, b: float = budget) -> bool:
                if a.action_type in {"Bid", "CounterBid", "Pay"}:
                    return a.price <= b
                if a.action_type == "Accept":
                    return s.p_ask <= b
                return True
            return Predicate(f"budget_constraint_<={budget}", fn, ctype, priority)

        # ---- latency_constraint -------------------------------------------
        if cid == "latency_constraint":
            # Explicit seconds in instruction take precedence over config default
            match = re.search(r'within\s+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)', instruction)
            default_secs = defn.get(
                "default_max_seconds",
                self._session_cfg.get("max_t_elapsed_seconds", 60.0),
            )
            max_secs = float(match.group(1)) if match else float(default_secs)
            if not override_active and not always and not any(kw in instruction for kw in keywords):
                return None
            def fn(s: State, a: Action, ms: float = max_secs) -> bool:
                return s.t_elapsed <= ms
            return Predicate(f"latency_constraint_<={max_secs}s", fn, ctype, priority)

        # ---- reputation_constraint ----------------------------------------
        if cid == "reputation_constraint":
            if not override_active and not always and not any(kw in instruction for kw in keywords):
                return None
            min_rep: float = defn.get("default_min_reputation", 0.7)
            def fn(s: State, a: Action, mr: float = min_rep) -> bool:
                return s.rep >= mr if a.action_type in {"Accept", "Pay", "Bid"} else True
            return Predicate(f"reputation_constraint_>={min_rep}", fn, ctype, priority)

        # ---- delivery_sla_constraint --------------------------------------
        if cid == "delivery_sla_constraint":
            if not override_active and not always and not any(kw in instruction for kw in keywords):
                return None
            max_days: int = defn.get("default_max_days", 5)
            # Placeholder: fully active once delivery_sla_days is added to State
            def fn(s: State, a: Action) -> bool:
                return True
            return Predicate(
                f"delivery_sla_constraint_<={max_days}d", fn, ctype, priority
            )

        # ---- vendor_trust_constraint (hard) --------------------------------
        if cid == "vendor_trust_constraint":
            if not override_active and not always and not any(kw in instruction for kw in keywords):
                return None
            # Placeholder: fully active once vendor_tier is added to State
            def fn(s: State, a: Action) -> bool:
                return True
            return Predicate("vendor_trust_constraint", fn, "hard", 0)

        # ---- compliance_constraint (hard) ----------------------------------
        if cid == "compliance_constraint":
            if not always:
                return None
            # Placeholder: fully active once compliance_status is added to State
            def fn(s: State, a: Action) -> bool:
                return True
            return Predicate("compliance_constraint", fn, "hard", 0)

        # ---- payment_rail_constraint (hard) --------------------------------
        if cid == "payment_rail_constraint":
            if not always:
                return None
            # Enforced structurally: Pay actions only carry allowed rails (see
            # MarketEnvironment._build_action_space + config.allowed_payment_rails)
            def fn(s: State, a: Action) -> bool:
                return True
            return Predicate("payment_rail_constraint", fn, "hard", 0)

        # ---- geo_restriction_constraint (hard) -----------------------------
        if cid == "geo_restriction_constraint":
            if not always:
                return None
            allowed_regions: List[str] = defn.get("allowed_regions", [])
            # Placeholder: fully active once geo_region is added to State
            def fn(s: State, a: Action, regions: List[str] = allowed_regions) -> bool:
                return True
            return Predicate("geo_restriction_constraint", fn, "hard", 0)

        return None   # unknown constraint id — skip silently
