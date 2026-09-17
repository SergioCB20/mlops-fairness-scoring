"""
QA Gate 3 (post-mitigación): valida que el modelo, tras el ajuste de
umbrales por ThresholdOptimizer, satisfaga el Contrato C3.
"""
import sys
from pathlib import Path

import pytest
import yaml
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, demographic_parity_ratio

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zona2_ci_training.mitigate import train_predict_and_mitigate

CONTRACT_PATH = Path("contracts/c3_equidad_modelo.yaml")


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def prediction_result():
    return train_predict_and_mitigate()


def test_statistical_parity_difference_mitigated(prediction_result, contract):
    threshold = contract["fairness_metrics"]["spd_threshold"]
    mf = MetricFrame(metrics=selection_rate, y_true=prediction_result["y_true"],
                      y_pred=prediction_result["y_pred"], sensitive_features=prediction_result["z_test"])
    spd = mf.difference()
    assert spd <= threshold, f"SPD (mitigado) = {spd:.4f} excede {threshold}.\n{mf.by_group}"
    print(f"\n✅ SPD mitigado = {spd:.4f}")


def test_equal_opportunity_difference_mitigated(prediction_result, contract):
    threshold = contract["fairness_metrics"]["eod_threshold"]
    mf = MetricFrame(metrics=true_positive_rate, y_true=prediction_result["y_true"],
                      y_pred=prediction_result["y_pred"], sensitive_features=prediction_result["z_test"])
    eod = mf.difference()
    assert eod <= threshold, f"EOD (mitigado) = {eod:.4f} excede {threshold}.\n{mf.by_group}"
    print(f"\n✅ EOD mitigado = {eod:.4f}")


def test_disparate_impact_mitigated(prediction_result, contract):
    di_min, di_max = contract["fairness_metrics"]["di_min"], contract["fairness_metrics"]["di_max"]
    di = demographic_parity_ratio(y_true=prediction_result["y_true"], y_pred=prediction_result["y_pred"],
                                    sensitive_features=prediction_result["z_test"])
    assert di_min <= di <= di_max, f"DI (mitigado) = {di:.4f} fuera de [{di_min}, {di_max}]."
    print(f"\n✅ DI mitigado = {di:.4f}")
