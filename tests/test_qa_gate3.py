"""
QA Gate 3: Equidad del Modelo (Contrato C3)
SPD, EOD y Disparate Impact calculados con Fairlearn sobre el conjunto de prueba.
"""
import sys
from pathlib import Path

import pytest
import yaml
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    demographic_parity_ratio,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zona2_ci_training.train import train_and_predict

CONTRACT_PATH = Path("contracts/c3_equidad_modelo.yaml")


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def prediction_result():
    return train_and_predict()


def test_statistical_parity_difference(prediction_result, contract):
    """SPD = |P(pred=1|Z=a) - P(pred=1|Z=b)| <= umbral (NO condiciona en Y)."""
    cfg = contract["fairness_metrics"]
    threshold = cfg["spd_threshold"]

    mf = MetricFrame(
        metrics=selection_rate,
        y_true=prediction_result["y_true"],
        y_pred=prediction_result["y_pred"],
        sensitive_features=prediction_result["z_test"],
    )
    spd = mf.difference()

    assert spd <= threshold, (
        f"SPD = {spd:.4f} excede el umbral del Contrato C3 ({threshold}).\n"
        f"Selection rate por grupo:\n{mf.by_group}"
    )
    print(f"\n✅ SPD = {spd:.4f} (umbral: {threshold})")


def test_equal_opportunity_difference(prediction_result, contract):
    """EOD = |TPR(Z=a) - TPR(Z=b)| <= umbral (SÍ condiciona en Y=1, vía TPR)."""
    cfg = contract["fairness_metrics"]
    threshold = cfg["eod_threshold"]

    mf = MetricFrame(
        metrics=true_positive_rate,
        y_true=prediction_result["y_true"],
        y_pred=prediction_result["y_pred"],
        sensitive_features=prediction_result["z_test"],
    )
    eod = mf.difference()

    assert eod <= threshold, (
        f"EOD = {eod:.4f} excede el umbral del Contrato C3 ({threshold}).\n"
        f"TPR por grupo:\n{mf.by_group}"
    )
    print(f"\n✅ EOD = {eod:.4f} (umbral: {threshold})")


def test_disparate_impact(prediction_result, contract):
    """DI = min(P(pred=1|Z)) / max(P(pred=1|Z)) dentro de [di_min, di_max]."""
    cfg = contract["fairness_metrics"]
    di_min, di_max = cfg["di_min"], cfg["di_max"]

    di = demographic_parity_ratio(
        y_true=prediction_result["y_true"],
        y_pred=prediction_result["y_pred"],
        sensitive_features=prediction_result["z_test"],
    )

    assert di_min <= di <= di_max, (
        f"DI = {di:.4f} fuera del rango del Contrato C3 [{di_min}, {di_max}]."
    )
    print(f"\n✅ DI = {di:.4f} (rango: [{di_min}, {di_max}])")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
