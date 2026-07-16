# tests/test_config_analysis.py
from scat.config import DEFAULT_CONFIG


def test_analysis_gains_contract_keys_without_losing_existing():
    a = DEFAULT_CONFIG["analysis"]
    assert a["model_type"] == "rf" and "annotate" in a and "visualize" in a  # existing keys intact
    assert a["primary_metric"] == "total_deposits"
    assert a["normalization"] == "per_image"
    assert a["confidence_threshold"] == 0.60
