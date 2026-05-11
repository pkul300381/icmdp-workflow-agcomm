"""
test_config_loader.py
---------------------
Unit tests for config_loader.load_config().
"""
import copy
import pytest
import yaml

from config_loader import load_config, _REQUIRED_TOP_LEVEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_cfg(tmp_path, data: dict) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return str(p)


VALID_MINIMAL = {
    "state_space": {"vendor_state": {}, "session_state": {}, "belief_prior": {}},
    "action_space": {"enabled_action_types": [], "bid_price_levels": [],
                     "default_payment_rail": "ISO_20022",
                     "allowed_payment_rails": [], "payment_rails": {}},
    "intent_constraints": {"definitions": {}, "parser": {}},
    "policies": {"solver": {}, "feasibility": {}, "dynamic_intent": {},
                 "negotiation": {}, "belief_update": {}},
    "probabilities": {"state_transitions": {}, "observation_model": {},
                      "market_model": {}, "intent_parsing": {}},
    "rewards": {},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_returns_dict_for_valid_file(self, tmp_path):
        path = write_cfg(tmp_path, VALID_MINIMAL)
        cfg = load_config(path)
        assert isinstance(cfg, dict)

    def test_all_required_sections_present_in_returned_dict(self, tmp_path):
        path = write_cfg(tmp_path, VALID_MINIMAL)
        cfg = load_config(path)
        for section in _REQUIRED_TOP_LEVEL:
            assert section in cfg, f"Section '{section}' missing from loaded config"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(str(tmp_path / "does_not_exist.yaml"))

    @pytest.mark.parametrize("missing_section", _REQUIRED_TOP_LEVEL)
    def test_missing_top_level_section_raises_value_error(self, tmp_path, missing_section):
        data = copy.deepcopy(VALID_MINIMAL)
        del data[missing_section]
        path = write_cfg(tmp_path, data)
        with pytest.raises(ValueError, match=missing_section):
            load_config(path)

    def test_malformed_yaml_raises_yaml_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("key: [\nunclosed bracket")
        with pytest.raises(yaml.YAMLError):
            load_config(str(p))

    def test_values_preserved_from_yaml(self, tmp_path):
        data = copy.deepcopy(VALID_MINIMAL)
        data["rewards"] = {"successful_pay_reward": 42.0}
        path = write_cfg(tmp_path, data)
        cfg = load_config(path)
        assert cfg["rewards"]["successful_pay_reward"] == 42.0

    def test_default_path_resolves_to_config_yaml(self, monkeypatch, tmp_path):
        """load_config() with no argument looks for 'config.yaml' in cwd."""
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(VALID_MINIMAL))
        monkeypatch.chdir(tmp_path)
        cfg = load_config()   # no argument — uses default "config.yaml"
        assert isinstance(cfg, dict)

    def test_nested_values_accessible(self, tmp_path):
        data = copy.deepcopy(VALID_MINIMAL)
        data["state_space"]["session_state"] = {"initial_budget": 500.0}
        path = write_cfg(tmp_path, data)
        cfg = load_config(path)
        assert cfg["state_space"]["session_state"]["initial_budget"] == 500.0

    def test_list_values_preserved(self, tmp_path):
        data = copy.deepcopy(VALID_MINIMAL)
        data["action_space"]["bid_price_levels"] = [50.0, 100.0, 200.0]
        path = write_cfg(tmp_path, data)
        cfg = load_config(path)
        assert cfg["action_space"]["bid_price_levels"] == [50.0, 100.0, 200.0]
