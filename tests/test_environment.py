"""
test_environment.py
-------------------
Unit tests for State, Action, make_full_state, MarketEnvironment, BeliefTracker.
"""
import copy
import random
import pytest

from environment import (
    Action, State, MarketEnvironment, BeliefTracker, make_full_state,
)
from tests.conftest import make_state, make_action


# ===========================================================================
# State
# ===========================================================================

class TestState:

    def test_q_key_excludes_t_elapsed(self):
        s1 = make_state(t_elapsed=0.0)
        s2 = make_state(t_elapsed=99.0)
        assert s1.q_key() == s2.q_key()

    def test_q_key_excludes_budget_spent(self):
        s1 = make_state(budget_spent=0.0)
        s2 = make_state(budget_spent=500.0)
        assert s1.q_key() == s2.q_key()

    def test_q_key_excludes_negotiation_round(self):
        s1 = make_state(negotiation_round=0)
        s2 = make_state(negotiation_round=10)
        assert s1.q_key() == s2.q_key()

    def test_q_key_differs_for_different_p_ask(self):
        s1 = make_state(p_ask=100.0)
        s2 = make_state(p_ask=200.0)
        assert s1.q_key() != s2.q_key()

    def test_q_key_differs_for_different_inv(self):
        s1 = make_state(inv=1)
        s2 = make_state(inv=0)
        assert s1.q_key() != s2.q_key()

    def test_q_key_differs_for_different_eco(self):
        s1 = make_state(eco=1)
        s2 = make_state(eco=0)
        assert s1.q_key() != s2.q_key()

    def test_q_key_differs_for_different_vendor(self):
        s1 = make_state(vendor="VendorA")
        s2 = make_state(vendor="VendorB")
        assert s1.q_key() != s2.q_key()

    def test_str_representation_includes_vendor(self):
        s = make_state(vendor="VendorX")
        assert "VendorX" in str(s)

    def test_str_representation_includes_ask_price(self):
        s = make_state(p_ask=150.0)
        assert "150" in str(s)

    def test_str_representation_includes_t_elapsed(self):
        s = make_state(t_elapsed=7.5)
        assert "7.5" in str(s)

    def test_state_is_hashable(self):
        s = make_state()
        d = {s: "value"}
        assert d[s] == "value"

    def test_state_default_session_fields_are_zero(self):
        s = State(p_ask=100.0, inv=1, eco=1, rep=0.9, vendor="V")
        assert s.t_elapsed == 0.0
        assert s.budget_spent == 0.0
        assert s.negotiation_round == 0


# ===========================================================================
# Action
# ===========================================================================

class TestAction:

    def test_str_query(self):
        a = make_action("Query", "VendorA")
        assert str(a) == "Query(VendorA)"

    def test_str_bid_includes_price(self):
        a = make_action("Bid", "VendorA", price=150.0)
        assert str(a) == "Bid(VendorA, $150.0)"

    def test_str_pay_includes_rail(self):
        a = Action("Pay", "VendorA", price=100.0, rail="ISO_20022")
        assert "ISO_20022" in str(a)
        assert "100.0" in str(a)

    def test_str_pay_without_rail(self):
        a = Action("Pay", "VendorA", price=100.0, rail="")
        result = str(a)
        assert "Pay(VendorA" in result
        assert "rail" not in result

    def test_str_counter_bid(self):
        a = make_action("CounterBid", "VendorA", price=200.0)
        assert "CounterBid" in str(a)
        assert "200.0" in str(a)

    def test_action_is_hashable(self):
        a = make_action("Query")
        s = {a}
        assert a in s

    def test_action_default_rail_is_empty(self):
        a = Action("Query", "VendorA")
        assert a.rail == ""


# ===========================================================================
# make_full_state
# ===========================================================================

class TestMakeFullState:

    def test_preserves_vendor_fields(self):
        vendor = make_state(p_ask=100.0, inv=1, eco=0, rep=0.7, vendor="V")
        full = make_full_state(vendor, t_elapsed=5.0, budget_spent=50.0, negotiation_round=2)
        assert full.p_ask == 100.0
        assert full.inv == 1
        assert full.eco == 0
        assert full.rep == 0.7
        assert full.vendor == "V"

    def test_injects_session_fields(self):
        vendor = make_state()
        full = make_full_state(vendor, t_elapsed=12.5, budget_spent=75.0, negotiation_round=3)
        assert full.t_elapsed == 12.5
        assert full.budget_spent == 75.0
        assert full.negotiation_round == 3

    def test_returns_new_state_object(self):
        vendor = make_state()
        full = make_full_state(vendor, t_elapsed=1.0, budget_spent=0.0, negotiation_round=0)
        assert full is not vendor


# ===========================================================================
# MarketEnvironment
# ===========================================================================

class TestMarketEnvironment:

    def test_init_creates_true_states_for_all_vendors(self, env, vendors):
        for v in vendors:
            assert v in env.true_states

    def test_action_space_contains_only_enabled_types(self, env, cfg):
        enabled = set(cfg["action_space"]["enabled_action_types"])
        for action in env.get_all_actions():
            assert action.action_type in enabled

    def test_bid_price_levels_match_config(self, env, cfg):
        levels = cfg["action_space"]["bid_price_levels"]
        bid_prices = {a.price for a in env.get_all_actions() if a.action_type == "Bid"}
        assert bid_prices == set(levels)

    def test_pay_action_carries_default_rail(self, env, cfg):
        default_rail = cfg["action_space"]["default_payment_rail"]
        pay_actions = [a for a in env.get_all_actions() if a.action_type == "Pay"]
        assert len(pay_actions) > 0
        for a in pay_actions:
            assert a.rail == default_rail

    def test_actions_cover_both_vendors(self, env, vendors):
        action_vendors = {a.vendor for a in env.get_all_actions()}
        for v in vendors:
            assert v in action_vendors

    def test_reset_returns_state_with_correct_vendor(self, env):
        s = env.reset("VendorA")
        assert s.vendor == "VendorA"

    def test_reset_returns_state_within_configured_p_ask_levels(self, env, cfg):
        levels = cfg["action_space"]["bid_price_levels"]
        for _ in range(20):
            s = env.reset("VendorA")
            assert s.p_ask in cfg["state_space"]["vendor_state"]["p_ask_levels"]

    def test_reset_state_has_no_session_fields(self, env):
        s = env.reset("VendorA")
        assert s.t_elapsed == 0.0
        assert s.budget_spent == 0.0
        assert s.negotiation_round == 0

    def test_step_query_returns_query_penalty(self, env, cfg):
        env.true_states["VendorA"] = make_state(vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("Query"))
        assert reward == cfg["rewards"]["query_step_penalty"]
        assert done is False

    def test_step_deep_query_returns_deep_query_penalty(self, env, cfg):
        env.true_states["VendorA"] = make_state(vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("DeepQuery"))
        assert reward == cfg["rewards"]["deep_query_step_penalty"]
        assert done is False

    def test_step_successful_bid_reward(self, env, cfg):
        env.true_states["VendorA"] = make_state(p_ask=100.0, vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("Bid", price=200.0))
        assert reward == cfg["rewards"]["successful_bid_reward"]
        assert done is False

    def test_step_failed_bid_penalty(self, env, cfg):
        env.true_states["VendorA"] = make_state(p_ask=300.0, vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("Bid", price=100.0))
        assert reward == cfg["rewards"]["failed_bid_penalty"]
        assert done is False

    def test_step_accept_reward(self, env, cfg):
        env.true_states["VendorA"] = make_state(vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("Accept"))
        assert reward == cfg["rewards"]["accept_reward"]
        assert done is False

    def test_step_reject_is_terminal(self, env, cfg):
        env.true_states["VendorA"] = make_state(vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("Reject"))
        assert reward == cfg["rewards"]["reject_reward"]
        assert done is True

    def test_step_successful_pay_reward_and_terminal(self, env, cfg):
        env.true_states["VendorA"] = make_state(p_ask=100.0, inv=1, vendor="VendorA")
        _, reward, done = env.step("VendorA", Action("Pay", "VendorA", 200.0, "ISO_20022"))
        assert reward == cfg["rewards"]["successful_pay_reward"]
        assert done is True

    def test_step_successful_pay_depletes_inventory(self, env):
        env.true_states["VendorA"] = make_state(p_ask=100.0, inv=1, vendor="VendorA")
        next_state, _, _ = env.step("VendorA", Action("Pay", "VendorA", 200.0, "ISO_20022"))
        assert env.true_states["VendorA"].inv == 0

    def test_step_failed_pay_insufficient_price(self, env, cfg):
        env.true_states["VendorA"] = make_state(p_ask=300.0, inv=1, vendor="VendorA")
        _, reward, done = env.step("VendorA", Action("Pay", "VendorA", 100.0, "ISO_20022"))
        assert reward == cfg["rewards"]["failed_pay_penalty"]
        assert done is True

    def test_step_failed_pay_no_inventory(self, env, cfg):
        env.true_states["VendorA"] = make_state(p_ask=100.0, inv=0, vendor="VendorA")
        _, reward, done = env.step("VendorA", Action("Pay", "VendorA", 200.0, "ISO_20022"))
        assert reward == cfg["rewards"]["failed_pay_penalty"]
        assert done is True

    def test_step_escalate_is_terminal(self, env, cfg):
        env.true_states["VendorA"] = make_state(vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("EscalateToHuman"))
        assert reward == cfg["rewards"]["escalate_penalty"]
        assert done is True

    def test_step_abandon_is_terminal(self, env, cfg):
        env.true_states["VendorA"] = make_state(vendor="VendorA")
        _, reward, done = env.step("VendorA", make_action("Abandon"))
        assert done is True

    def test_state_transitions_occur_when_drift_prob_is_1(self, cfg, vendors):
        """With price_drift_prob=1.0 every non-terminal step changes the state."""
        cfg["probabilities"]["state_transitions"]["price_drift_prob"] = 1.0
        cfg["probabilities"]["state_transitions"]["price_drift_direction"] = "up_only"
        cfg["probabilities"]["state_transitions"]["price_drift_magnitude"] = 100.0
        cfg["state_space"]["vendor_state"]["p_ask_levels"] = [100.0, 200.0, 300.0, 400.0, 500.0]
        env2 = MarketEnvironment(vendors, cfg)
        env2.true_states["VendorA"] = make_state(p_ask=100.0, vendor="VendorA")
        before = env2.true_states["VendorA"].p_ask
        env2.step("VendorA", make_action("Query"))
        after = env2.true_states["VendorA"].p_ask
        assert after != before   # price must have drifted

    def test_no_transitions_when_drift_prob_is_0(self, env):
        """Deterministic config sets price_drift_prob=0 — state unchanged after query."""
        env.true_states["VendorA"] = make_state(p_ask=200.0, inv=1, eco=1, rep=0.9,
                                                 vendor="VendorA")
        before = env.true_states["VendorA"]
        env.step("VendorA", make_action("Query"))
        after = env.true_states["VendorA"]
        assert before.p_ask == after.p_ask
        assert before.inv == after.inv

    def test_get_payment_rail_catalog_has_all_categories(self, env, cfg):
        catalog = env.get_payment_rail_catalog()
        categories = {r["category"] for r in catalog}
        expected = set(cfg["action_space"]["payment_rails"].keys())
        assert categories == expected

    def test_get_payment_rail_catalog_each_entry_has_id(self, env):
        for entry in env.get_payment_rail_catalog():
            assert "id" in entry
            assert "description" in entry
            assert "avg_settlement_seconds" in entry
            assert "cost_pct" in entry


# ===========================================================================
# BeliefTracker
# ===========================================================================

class TestBeliefTracker:

    def test_initial_belief_matches_prior(self, env, cfg):
        tracker = BeliefTracker(env, cfg)
        prior = cfg["state_space"]["belief_prior"]
        for v in env.vendors:
            s = tracker.get_most_likely_state(v)
            assert s.p_ask == prior["p_ask"]
            assert s.inv == prior["inv"]
            assert s.eco == prior["eco"]
            assert abs(s.rep - prior["rep"]) < 1e-9

    def test_perfect_observation_updates_belief_exactly(self, env, cfg):
        tracker = BeliefTracker(env, cfg)
        observed = make_state(p_ask=150.0, inv=0, eco=0, rep=0.6, vendor="VendorA")
        tracker.update_belief(make_action("Query"), observed, "VendorA")
        updated = tracker.get_most_likely_state("VendorA")
        assert updated.p_ask == 150.0
        assert updated.inv == 0
        assert updated.eco == 0

    def test_non_state_observation_ignored(self, env, cfg):
        tracker = BeliefTracker(env, cfg)
        prior = tracker.get_most_likely_state("VendorA")
        tracker.update_belief(make_action("Query"), "not_a_state", "VendorA")
        assert tracker.get_most_likely_state("VendorA") == prior

    def test_noisy_observation_may_differ_from_true(self, cfg, vendors):
        """With high noise rates the tracked belief may differ from true state."""
        cfg["probabilities"]["observation_model"]["query_reveals_true_state"] = False
        cfg["probabilities"]["observation_model"]["eco_cert_false_positive_rate"] = 1.0
        cfg["policies"]["belief_update"]["observation_noise_model"] = "gaussian"
        env2 = MarketEnvironment(vendors, cfg)
        tracker = BeliefTracker(env2, cfg)
        true_state = make_state(eco=0, vendor="VendorA")
        tracker.update_belief(make_action("Query"), true_state, "VendorA")
        # eco_cert_false_positive_rate=1.0 → must observe eco=1 when true eco=0
        assert tracker.get_most_likely_state("VendorA").eco == 1

    def test_belief_tracker_has_entry_for_every_vendor(self, env, cfg):
        tracker = BeliefTracker(env, cfg)
        for v in env.vendors:
            s = tracker.get_most_likely_state(v)
            assert s.vendor == v
