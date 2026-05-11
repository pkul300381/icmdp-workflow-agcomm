"""
conftest.py
-----------
Shared fixtures for the ICMDP test suite.

MINIMAL_CFG is a self-contained config dict that mirrors the full config.yaml
structure but uses deterministic values (zero drift probabilities, zero latency
penalty, greedy solver) so that tests produce reproducible results without
depending on any external file.
"""
import copy
import sys
import os

# Ensure the project root is on sys.path so all modules import cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from environment import MarketEnvironment, State, Action
from intent_parser import LLMIntentParser
from feasibility import FeasibilityMonitor
from solvers import MaskedQLearning
from execution_engine import ExecutionEngine


# ---------------------------------------------------------------------------
# Minimal, fully-deterministic configuration dict
# ---------------------------------------------------------------------------
MINIMAL_CFG: dict = {
    "state_space": {
        "vendor_state": {
            "p_ask_levels": [100.0, 200.0, 300.0],
            "inv_high_prob_weight": 3,
            "eco_options": [0, 1],
            "reputation_range": [0.5, 1.0],
        },
        "session_state": {
            "initial_budget": 200.0,
            "max_t_elapsed_seconds": 60.0,
            "step_duration_seconds": 1.0,
            "max_items_to_purchase": 1,
        },
        "belief_prior": {
            "p_ask": 200.0,
            "inv": 1,
            "eco": 1,
            "rep": 0.8,
        },
    },
    "action_space": {
        "enabled_action_types": [
            "Query", "DeepQuery", "Bid", "CounterBid",
            "Accept", "Reject", "Pay",
            "EscalateToHuman", "Abandon", "SwitchVendor",
            "VerifyCompliance", "RequestCertification", "RequestDiscount",
        ],
        "bid_price_levels": [100.0, 200.0, 300.0],
        "default_payment_rail": "ISO_20022",
        "allowed_payment_rails": ["ISO_20022", "ACH"],
        "payment_rails": {
            "traditional": [
                {
                    "id": "ISO_20022",
                    "description": "ISO 20022 messaging",
                    "avg_settlement_seconds": 3600,
                    "cost_pct": 0.002,
                    "requires_compliance": ["iso20022_compliant"],
                },
                {
                    "id": "ACH",
                    "description": "Automated Clearing House",
                    "avg_settlement_seconds": 86400,
                    "cost_pct": 0.001,
                    "requires_compliance": [],
                },
            ],
            "stablecoin": [
                {
                    "id": "USDC",
                    "description": "USD Coin stablecoin",
                    "avg_settlement_seconds": 15,
                    "cost_pct": 0.0005,
                    "requires_compliance": [],
                },
            ],
        },
    },
    "intent_constraints": {
        "definitions": {
            "eco_constraint": {
                "type": "soft",
                "relaxation_priority": 1,
                "always_active": False,
                "keywords": ["carbon-neutral", "eco-friendly", "green", "sustainable"],
            },
            "budget_constraint": {
                "type": "soft",
                "relaxation_priority": 2,
                "always_active": False,
                "keywords": ["max", "under", "no more than", "budget", "spending", "spend"],
            },
            "latency_constraint": {
                "type": "soft",
                "relaxation_priority": 3,
                "always_active": False,
                "keywords": ["within", "seconds", "second"],
                "default_max_seconds": 60.0,
            },
            "reputation_constraint": {
                "type": "soft",
                "relaxation_priority": 4,
                "always_active": False,
                "keywords": ["trusted", "reliable", "reputable"],
                "default_min_reputation": 0.7,
            },
            "inventory_constraint": {
                "type": "hard",
                "relaxation_priority": 0,
                "always_active": True,
            },
            "compliance_constraint": {
                "type": "hard",
                "relaxation_priority": 0,
                "always_active": True,
                "required_status": "iso20022_compliant",
            },
            "payment_rail_constraint": {
                "type": "hard",
                "relaxation_priority": 0,
                "always_active": True,
            },
            "geo_restriction_constraint": {
                "type": "hard",
                "relaxation_priority": 0,
                "always_active": True,
                "allowed_regions": ["US", "EU"],
            },
        },
        "parser": {
            "backend": "regex",
            "confidence_threshold": 0.75,
            "ambiguity_resolution": "most_conservative",
            "re_parse_on_state_change": False,
        },
    },
    "policies": {
        "solver": {
            "type": "masked_q_learning",
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.0,        # fully greedy for deterministic tests
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "training_episodes": 5,
            "max_steps_per_episode": 5,
            "convergence_threshold": 0.001,
        },
        "feasibility": {
            "max_relaxations": 2,
            "infeasibility_response": "relax",
            "relaxation_cooldown_steps": 0,
        },
        "dynamic_intent": {
            "update_trigger": "never",
            "update_interval_steps": 10,
            "constraint_injection_mode": "full_reparse",
        },
        "negotiation": {
            "initial_bid_strategy": "ask_price",
            "bid_increment": 25.0,
            "max_negotiation_rounds": 5,
            "vendor_selection_strategy": "greedy_q",
            "multi_vendor_mode": "sequential",
        },
        "belief_update": {
            "mode": "argmax",
            "observation_noise_model": "perfect",
        },
    },
    "probabilities": {
        "state_transitions": {
            "price_drift_prob": 0.0,       # no drift → deterministic transitions
            "price_drift_direction": "symmetric",
            "price_drift_magnitude": 50.0,
            "inventory_depletion_prob": 0.0,
            "inventory_restock_prob": 0.0,
            "eco_cert_revocation_prob": 0.0,
            "reputation_decay_rate": 0.0,
        },
        "observation_model": {
            "query_reveals_true_state": True,
            "price_observation_accuracy": 1.0,
            "inventory_signal_accuracy": 1.0,
            "eco_cert_false_positive_rate": 0.0,
            "eco_cert_false_negative_rate": 0.0,
            "reputation_observation_variance": 0.0,
        },
        "market_model": {
            "vendor_acceptance_margin": 0.0,
            "stockout_prob_per_step": 0.0,
            "price_drop_prob_per_step": 0.0,
            "bid_counter_offer_prob": 0.0,
        },
        "intent_parsing": {
            "constraint_type_priors": {
                "eco_constraint": 0.3,
                "budget_constraint": 0.8,
            },
            "parse_error_rate": 0.0,
        },
    },
    "rewards": {
        "query_step_penalty": -1.0,
        "deep_query_step_penalty": -2.0,
        "successful_bid_reward": 10.0,
        "failed_bid_penalty": -5.0,
        "accept_reward": 5.0,
        "reject_reward": 0.0,
        "successful_pay_reward": 100.0,
        "failed_pay_penalty": -50.0,
        "latency_penalty_per_second": 0.0,   # disabled for clean reward assertions
        "escalate_penalty": -10.0,
        "constraint_violation_penalty": -1000.0,
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> dict:
    """Fresh deep copy of MINIMAL_CFG for each test."""
    return copy.deepcopy(MINIMAL_CFG)


@pytest.fixture
def vendors():
    return ["VendorA", "VendorB"]


@pytest.fixture
def env(vendors, cfg):
    return MarketEnvironment(vendors, cfg)


@pytest.fixture
def parser(cfg):
    return LLMIntentParser(cfg)


@pytest.fixture
def monitor(cfg):
    return FeasibilityMonitor(cfg)


@pytest.fixture
def agent(cfg):
    return MaskedQLearning(cfg)


@pytest.fixture
def engine(env, parser, monitor, agent, cfg):
    return ExecutionEngine(env, parser, monitor, agent, cfg)


# ---------------------------------------------------------------------------
# Convenience state builders (deterministic — no random sampling)
# ---------------------------------------------------------------------------

def make_state(
    vendor="VendorA",
    p_ask=100.0,
    inv=1,
    eco=1,
    rep=0.9,
    t_elapsed=0.0,
    budget_spent=0.0,
    negotiation_round=0,
) -> State:
    return State(
        vendor=vendor,
        p_ask=p_ask,
        inv=inv,
        eco=eco,
        rep=rep,
        t_elapsed=t_elapsed,
        budget_spent=budget_spent,
        negotiation_round=negotiation_round,
    )


def make_action(action_type="Query", vendor="VendorA", price=0.0, rail="") -> Action:
    return Action(action_type=action_type, vendor=vendor, price=price, rail=rail)
