"""
main.py
-------
Entry point for the ICMDP Agentic Commerce demo.

All parameters are read from config.yaml.  To change any behaviour
(state space, action space, constraints, RL hyperparameters, payment rails,
probabilities, rewards) edit config.yaml — no code change needed.
"""
import json

from config_loader import load_config
from environment import MarketEnvironment
from intent_parser import LLMIntentParser
from feasibility import FeasibilityMonitor
from solvers import MaskedQLearning
from execution_engine import ExecutionEngine


def run_demo() -> None:
    cfg = load_config("config.yaml")

    print("Initializing Agentic Commerce Environment...")
    vendors = ["VendorA", "VendorB"]

    env     = MarketEnvironment(vendors, cfg)
    parser  = LLMIntentParser(cfg)
    monitor = FeasibilityMonitor(cfg)
    agent   = MaskedQLearning(cfg)
    engine  = ExecutionEngine(env, parser, monitor, agent, cfg)

    instruction   = ("Purchase 100 units from a carbon-neutral supplier, "
                     "spending no more than $200")
    target_vendor = "VendorA"

    # ---- Phase 1: online training ----------------------------------------
    print("\n[Phase 1] Online Learning...")
    engine.train_online(instruction, target_vendor)

    # ---- Phase 2: greedy execution + certificate -------------------------
    print("\n[Phase 2] Execution & Audit...")
    cert = engine.execute_transaction(instruction, target_vendor)

    print("\n--- Certificate of Execution ---")
    print(json.dumps(cert, indent=2))

    with open("execution_certificate.json", "w") as f:
        json.dump(cert, f, indent=2)
    print("\nCertificate saved to execution_certificate.json")

    # ---- Payment rail catalogue (from config) ----------------------------
    print("\n--- Payment Rail Catalogue ---")
    for rail in env.get_payment_rail_catalog():
        print(
            f"  [{rail['category']:18s}] {rail['id']:22s}  "
            f"settlement={rail['avg_settlement_seconds']:>6}s  "
            f"cost={rail['cost_pct']*100:.4f}%  "
            f"{rail['description']}"
        )


if __name__ == "__main__":
    run_demo()
