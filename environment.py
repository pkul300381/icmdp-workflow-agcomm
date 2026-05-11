"""
environment.py
--------------
Defines the ICMDP state space (S), action space (A), stochastic transition
model P(s'|s,a), POMDP observation model Z(o|s',a), and belief tracker.

All parameters are driven by config.yaml — no hardcoded values.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import random


# =============================================================================
# Core MDP types
# =============================================================================

@dataclass(frozen=True)
class Action:
    """
    A member of the action space A.

    The `rail` field carries the payment-rail identifier for Pay actions
    (logged in the certificate of execution). It does not affect the Q-table
    key because it is determined by policy config, not learned.
    """
    action_type: str   # e.g. "Query", "Bid", "Pay", "Accept", …
    vendor: str
    price: float = 0.0
    rail: str = ""     # payment rail for Pay actions (config: default_payment_rail)

    def __str__(self) -> str:
        if self.action_type == "Pay":
            rail_str = f", rail={self.rail}" if self.rail else ""
            return f"Pay({self.vendor}, ${self.price}{rail_str})"
        if self.action_type in ("Bid", "CounterBid", "RequestDiscount"):
            return f"{self.action_type}({self.vendor}, ${self.price})"
        return f"{self.action_type}({self.vendor})"


@dataclass(frozen=True)
class State:
    """
    A member of the state space S = vendor_state × session_state.

    Vendor state (p_ask, inv, eco, rep) represents the hidden market state
    per paper §6.2.  Session state (t_elapsed, budget_spent, negotiation_round)
    is required for latency and budget intent constraints (paper §6.3).

    q_key() strips session dims so the Q-table generalises across episodes
    that share the same vendor situation but different elapsed times.
    """
    # Vendor state (paper §6.2)
    p_ask: float        # seller's posted ask price (discretised)
    inv: int            # inventory: 1 = available, 0 = out of stock
    eco: int            # eco-certified: 1 = yes, 0 = no
    rep: float          # reputation score ∈ [0, 1]
    vendor: str

    # Session state (paper §6.3 — needed by latency and budget constraints)
    t_elapsed: float = 0.0          # simulated seconds elapsed in this episode
    budget_spent: float = 0.0       # cumulative spend in this session
    negotiation_round: int = 0      # step counter for this vendor interaction

    def q_key(self) -> str:
        """
        Vendor-only Q-table key.  Excludes session dims to enable cross-episode
        generalisation (a vendor at ask=$150, inv=1, eco=1, rep=0.8 is the same
        Q-table entry whether it appears at t=2s or t=15s).
        """
        return (f"v={self.vendor}|ask={int(self.p_ask)}"
                f"|inv={self.inv}|eco={self.eco}|rep={self.rep:.2f}")

    def __str__(self) -> str:
        return (f"State(vendor={self.vendor}, ask=${self.p_ask}, "
                f"inv={self.inv}, eco={self.eco}, rep={self.rep:.2f}, "
                f"t={self.t_elapsed:.1f}s, spent=${self.budget_spent:.2f})")


def make_full_state(
    vendor_state: State,
    t_elapsed: float,
    budget_spent: float,
    negotiation_round: int,
) -> State:
    """
    Combine a vendor-only State with session-level tracking fields to produce
    the full ICMDP state used for constraint evaluation.

    The execution engine calls this after every step to attach up-to-date
    session accounting before passing the state to the feasibility monitor.
    """
    return State(
        p_ask=vendor_state.p_ask,
        inv=vendor_state.inv,
        eco=vendor_state.eco,
        rep=vendor_state.rep,
        vendor=vendor_state.vendor,
        t_elapsed=t_elapsed,
        budget_spent=budget_spent,
        negotiation_round=negotiation_round,
    )


# =============================================================================
# Market environment  (P : S × A × S → [0,1])
# =============================================================================

class MarketEnvironment:
    """
    Stochastic market environment.  Implements the transition function
    P(s'|s,a) and the reward function R(s,a) as defined in config.yaml.

    Partial observability is handled by BeliefTracker (below); this class
    manages the *true* hidden states.
    """

    def __init__(self, vendors: List[str], cfg: dict) -> None:
        self.vendors = vendors
        self._cfg = cfg

        vs = cfg["state_space"]["vendor_state"]
        self._p_ask_levels: List[float] = vs["p_ask_levels"]
        self._rep_range: List[float] = vs["reputation_range"]
        self._eco_options: List[int] = vs.get("eco_options", [0, 1])
        inv_weight = vs.get("inv_high_prob_weight", 3)
        # inv=1 appears `inv_weight` times, inv=0 appears once → P(inv=1) = weight/(weight+1)
        self._inv_choices: List[int] = [0] + [1] * inv_weight

        self._rewards: dict = cfg["rewards"]
        self._trans: dict = cfg["probabilities"]["state_transitions"]

        ac = cfg["action_space"]
        self._bid_levels: List[float] = ac["bid_price_levels"]
        self._enabled: set = set(ac["enabled_action_types"])
        self._default_rail: str = ac["default_payment_rail"]

        self.true_states: Dict[str, State] = {v: self._sample_vendor(v) for v in vendors}
        self.all_actions: List[Action] = self._build_action_space()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_vendor(self, vendor: str) -> State:
        return State(
            vendor=vendor,
            p_ask=random.choice(self._p_ask_levels),
            inv=random.choice(self._inv_choices),
            eco=random.choice(self._eco_options),
            rep=random.uniform(*self._rep_range),
        )

    def _build_action_space(self) -> List[Action]:
        """
        Constructs A from config.  Only action types listed in
        `enabled_action_types` are included.  Pay actions carry the
        `default_payment_rail` so the rail is logged in the certificate
        without expanding the training action space.
        """
        actions: List[Action] = []
        e = self._enabled
        no_price = [
            "Query", "DeepQuery", "Accept", "Reject",
            "VerifyCompliance", "RequestCertification",
            "EscalateToHuman", "Abandon", "SwitchVendor",
        ]
        priced = ["Bid", "CounterBid", "RequestDiscount"]

        for v in self.vendors:
            for atype in no_price:
                if atype in e:
                    actions.append(Action(atype, v))
            for p in self._bid_levels:
                for atype in priced:
                    if atype in e:
                        actions.append(Action(atype, v, p))
                if "Pay" in e:
                    actions.append(Action("Pay", v, p, self._default_rail))
        return actions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_all_actions(self) -> List[Action]:
        return self.all_actions

    def get_payment_rail_catalog(self) -> List[Dict]:
        """
        Returns the full payment rail catalogue from config as a flat list,
        enriched with the category name.  Useful for certificate logging
        and UI display.
        """
        catalog: List[Dict] = []
        for category, rails in self._cfg["action_space"]["payment_rails"].items():
            for r in rails:
                catalog.append({**r, "category": category})
        return catalog

    def reset(self, vendor: str) -> State:
        """Reset vendor to a freshly sampled true state (starts a new episode)."""
        self.true_states[vendor] = self._sample_vendor(vendor)
        return self.true_states[vendor]

    def step(
        self, vendor: str, action: Action, t_elapsed: float = 0.0
    ) -> Tuple[State, float, bool]:
        """
        Execute *action* against *vendor*.  Returns (next_vendor_state, reward, done).

        Session-level fields (t_elapsed, budget_spent) are NOT stored here;
        the execution engine owns that accounting and injects them via
        make_full_state() when assembling the full ICMDP state.
        """
        ts = self.true_states[vendor]
        r = self._rewards
        trans = self._trans
        reward = 0.0
        done = False

        if action.action_type == "Query":
            reward = r["query_step_penalty"]

        elif action.action_type == "DeepQuery":
            reward = r["deep_query_step_penalty"]

        elif action.action_type in ("Bid", "CounterBid"):
            reward = (r["successful_bid_reward"]
                      if action.price >= ts.p_ask
                      else r["failed_bid_penalty"])

        elif action.action_type == "Accept":
            reward = r["accept_reward"]

        elif action.action_type == "Reject":
            reward = r["reject_reward"]
            done = True

        elif action.action_type == "Pay":
            if action.price >= ts.p_ask and ts.inv == 1:
                reward = r["successful_pay_reward"]
                done = True
                # Consume inventory
                self.true_states[vendor] = State(
                    ts.p_ask, 0, ts.eco, ts.rep, vendor
                )
            else:
                reward = r["failed_pay_penalty"]
                done = True

        elif action.action_type in ("VerifyCompliance", "RequestCertification"):
            reward = r["query_step_penalty"]

        elif action.action_type == "EscalateToHuman":
            reward = r.get("escalate_penalty", -10.0)
            done = True

        elif action.action_type in ("Abandon", "SwitchVendor"):
            reward = r["reject_reward"]
            done = True

        elif action.action_type == "RequestDiscount":
            reward = r["query_step_penalty"]   # small cost; discount outcome is future work

        # Per-step latency penalty (set latency_penalty_per_second: 0.0 to disable)
        reward += r.get("latency_penalty_per_second", 0.0) * t_elapsed

        # Stochastic state transitions P(s'|s,a) — driven entirely by config
        if not done and random.random() < trans["price_drift_prob"]:
            direction = trans.get("price_drift_direction", "symmetric")
            magnitude = trans.get("price_drift_magnitude", 50.0)

            if direction == "up_only":
                candidate = ts.p_ask + magnitude
                new_p = min(self._p_ask_levels, key=lambda x: abs(x - candidate))
            elif direction == "down_only":
                candidate = ts.p_ask - magnitude
                new_p = min(self._p_ask_levels, key=lambda x: abs(x - candidate))
            else:   # symmetric: jump to any price bin
                new_p = random.choice(self._p_ask_levels)

            new_inv = ts.inv
            if ts.inv == 1 and random.random() < trans["inventory_depletion_prob"]:
                new_inv = 0
            elif ts.inv == 0 and random.random() < trans["inventory_restock_prob"]:
                new_inv = 1

            new_eco = ts.eco
            if ts.eco == 1 and random.random() < trans["eco_cert_revocation_prob"]:
                new_eco = 0

            new_rep = max(0.0, min(1.0,
                ts.rep + random.gauss(0, trans["reputation_decay_rate"])))

            self.true_states[vendor] = State(new_p, new_inv, new_eco, new_rep, vendor)

        return self.true_states[vendor], reward, done


# =============================================================================
# Belief tracker  (POMDP layer — Z : S × A × O → [0,1])
# =============================================================================

class BeliefTracker:
    """
    Maintains the agent's belief b_t(s) over hidden vendor states (paper §3.2).

    The current implementation supports two modes driven by config:
      • argmax / query_reveals_true_state=true  — simplification used in §6.4 demo
      • gaussian                                 — applies configurable observation noise
    Full particle-filter or distribution-based belief updating (Eq. 6) is
    marked for future work.
    """

    def __init__(self, env: MarketEnvironment, cfg: dict) -> None:
        prior = cfg["state_space"]["belief_prior"]
        self._obs_cfg: dict = cfg["probabilities"]["observation_model"]
        self._mode: str = cfg["policies"]["belief_update"]["mode"]
        self._noise_model: str = cfg["policies"]["belief_update"]["observation_noise_model"]

        self.estimated_states: Dict[str, State] = {}
        for v in env.vendors:
            self.estimated_states[v] = State(
                vendor=v,
                p_ask=float(prior["p_ask"]),
                inv=int(prior["inv"]),
                eco=int(prior["eco"]),
                rep=float(prior["rep"]),
            )

    def update_belief(self, action: Action, observation: Any, vendor: str) -> None:
        """
        Update the belief estimate for *vendor* given *observation*.

        When `query_reveals_true_state` is true (config default), the
        observation is accepted as ground truth (perfect observability
        simplification from paper §6.4).  Otherwise, configurable noise
        is applied to simulate the POMDP observation model Z(o|s',a).
        """
        if not isinstance(observation, State):
            return

        obs_cfg = self._obs_cfg

        if obs_cfg.get("query_reveals_true_state", True) or self._noise_model == "perfect":
            # Perfect observation — simplification from §6.4
            self.estimated_states[vendor] = observation
            return

        # Gaussian / categorical observation noise (noise_model != perfect)
        inv_acc = obs_cfg.get("inventory_signal_accuracy", 0.95)
        eco_fp = obs_cfg.get("eco_cert_false_positive_rate", 0.02)
        eco_fn = obs_cfg.get("eco_cert_false_negative_rate", 0.01)
        rep_var = obs_cfg.get("reputation_observation_variance", 0.05)

        # Inventory: flip with probability (1 - accuracy)
        observed_inv = observation.inv if random.random() < inv_acc else (1 - observation.inv)

        # Eco: apply false-positive / false-negative rates
        if observation.eco == 1:
            observed_eco = 0 if random.random() < eco_fn else 1
        else:
            observed_eco = 1 if random.random() < eco_fp else 0

        # Reputation: additive Gaussian noise
        observed_rep = max(0.0, min(1.0,
            observation.rep + random.gauss(0, rep_var)))

        self.estimated_states[vendor] = State(
            vendor=vendor,
            p_ask=observation.p_ask,   # price assumed directly observable
            inv=observed_inv,
            eco=observed_eco,
            rep=observed_rep,
        )

    def get_most_likely_state(self, vendor: str) -> State:
        """Returns the argmax belief state (current mode for all solver types)."""
        return self.estimated_states[vendor]
