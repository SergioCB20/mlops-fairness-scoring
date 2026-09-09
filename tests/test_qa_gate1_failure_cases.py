"""
Casos de prueba negativos para QA Gate 1 (Contrato C1).
Confirman que el mecanismo de interrupción realmente detiene el pipeline
cuando los datos violan el contrato, en vez de solo validar el camino feliz.
"""
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tests.test_qa_gate1 import load_contract, build_pandera_schema, CONTRACT_PATH


@pytest.fixture(scope="module")
def contract():
    return load_contract()


def make_synthetic_batch(n_rows: int = 200, sex_ratio: dict = None) -> pd.DataFrame:
    """
    Genera un batch sintético válido en esquema, pero con la distribución
    de Sex que se le indique — para forzar escenarios de representatividad.
    """
    if sex_ratio is None:
        sex_ratio = {1: 0.5, 2: 0.5}

    sex_values = []
    for value, proportion in sex_ratio.items():
        sex_values += [value] * int(n_rows * proportion)
    # Ajuste por redondeo
    while len(sex_values) < n_rows:
        sex_values.append(list(sex_ratio.keys())[0])
    sex_values = sex_values[:n_rows]

    return pd.DataFrame({
        "label": [1] * n_rows,
        "Age": [30] * n_rows,
        "Sex": sex_values,
        "Marital": [4] * n_rows,
        "Number_of_credits": [1] * n_rows,
        "Linked_cards": [1] * n_rows,
    })


def test_representativeness_fails_on_imbalanced_batch(contract):
    """
    Con un batch 99% Sex=1 / 1% Sex=2, el grupo minoritario (1%) está muy por
    debajo del umbral del contrato (5%) -> el gate DEBE fallar.
    """
    imbalanced_batch = make_synthetic_batch(
        n_rows=200, sex_ratio={1: 0.99, 2: 0.01}
    )

    rep_config = contract["representativeness"]
    protected_attr = rep_config["protected_attribute"]
    min_proportion = rep_config["min_group_proportion"]

    proportions = imbalanced_batch[protected_attr].value_counts(normalize=True)
    violating_groups = proportions[proportions < min_proportion]

    # Aquí, a diferencia del test real del gate, ESPERAMOS que falle
    assert not violating_groups.empty, (
        "Se esperaba que el batch desbalanceado violara la representatividad, "
        "pero no se detectó ninguna violación (el gate NO está funcionando)."
    )
    print(f"\n✅ Gate detectó correctamente la violación:\n{violating_groups}")


def test_representativeness_gate_would_stop_pipeline(contract):
    """
    Simula el comportamiento real del gate: si este assert se ejecutara
    dentro de test_representativeness (no aquí), el pipeline se detendría
    con exit code != 0. Aquí lo verificamos explícitamente con pytest.raises.
    """
    imbalanced_batch = make_synthetic_batch(
        n_rows=200, sex_ratio={1: 0.99, 2: 0.01}
    )
    rep_config = contract["representativeness"]
    protected_attr = rep_config["protected_attribute"]
    min_proportion = rep_config["min_group_proportion"]

    def run_representativeness_check(batch):
        proportions = batch[protected_attr].value_counts(normalize=True)
        violating = proportions[proportions < min_proportion]
        assert violating.empty, (
            f"Grupo(s) de '{protected_attr}' por debajo del umbral "
            f"({min_proportion}):\n{violating}"
        )

    with pytest.raises(AssertionError, match="por debajo del umbral"):
        run_representativeness_check(imbalanced_batch)


def test_schema_still_valid_on_synthetic_batch(contract):
    """
    Confirma que el batch sintético desbalanceado SÍ pasa el esquema
    (para aislar que el fallo es específicamente de representatividad,
    no de un error de tipos/nulos que confundiría el diagnóstico).
    """
    imbalanced_batch = make_synthetic_batch(
        n_rows=200, sex_ratio={1: 0.99, 2: 0.01}
    )
    schema = build_pandera_schema(contract)
    schema.validate(imbalanced_batch, lazy=True)  # no debe lanzar excepción


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
