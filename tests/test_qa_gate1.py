"""
QA Gate 1: Validación de Esquema/Integridad y Representatividad (Contrato C1)
"""
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zona1_ingesta.ingestor import load_raw_batch, drop_leakage_columns

CONTRACT_PATH = Path("contracts/c1_representatividad.yaml")

PANDERA_DTYPE_MAP = {
    "int": pa.Int,
    "float": pa.Float,
    "str": pa.String,
}


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pandera_schema(contract: dict) -> pa.DataFrameSchema:
    """Traduce el bloque 'schema' del YAML a un DataFrameSchema de Pandera."""
    columns = {}
    for col in contract["schema"]["required_columns"]:
        checks = []
        if "allowed_values" in col:
            checks.append(pa.Check.isin(col["allowed_values"]))
        if "min_value" in col:
            checks.append(pa.Check.ge(col["min_value"]))
        if "max_value" in col:
            checks.append(pa.Check.le(col["max_value"]))

        columns[col["name"]] = pa.Column(
            PANDERA_DTYPE_MAP[col["dtype"]],
            checks=checks,
            nullable=col.get("nullable", False),
        )
    return pa.DataFrameSchema(columns, strict=False)


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def batch():
    df, batch_hash = load_raw_batch()
    df = drop_leakage_columns(df)
    return df


def test_schema_and_integrity(batch, contract):
    """Valida tipos, nulos y rangos declarados en el Contrato C1."""
    schema = build_pandera_schema(contract)
    try:
        schema.validate(batch, lazy=True)
    except pa.errors.SchemaErrors as err:
        pytest.fail(f"Violaciones de esquema/integridad:\n{err.failure_cases}")


def test_null_ratio(batch, contract):
    """Verifica que ninguna columna supere el ratio máximo de nulos permitido."""
    max_ratio = contract["integrity"]["max_null_ratio_per_column"]
    null_ratios = batch.isnull().mean()
    violating = null_ratios[null_ratios > max_ratio]
    assert violating.empty, (
        f"Columnas exceden el ratio máximo de nulos ({max_ratio}):\n{violating}"
    )


def test_duplicate_ratio(batch, contract):
    """Verifica que el ratio de filas duplicadas no exceda el máximo permitido."""
    max_ratio = contract["integrity"]["max_duplicate_rows_ratio"]
    dup_ratio = batch.duplicated().sum() / len(batch)
    assert dup_ratio <= max_ratio, (
        f"Ratio de duplicados ({dup_ratio:.4f}) excede el máximo ({max_ratio})"
    )


def test_representativeness(batch, contract):
    """
    QA Gate 1 - Representatividad: verifica que cada valor del atributo
    protegido tenga al menos la proporción mínima definida en el contrato.
    Fórmula: P(Z=z) >= min_group_proportion, para todo z en dominio(Z).
    """
    rep_config = contract["representativeness"]
    protected_attr = rep_config["protected_attribute"]
    min_proportion = rep_config["min_group_proportion"]

    proportions = batch[protected_attr].value_counts(normalize=True)
    violating_groups = proportions[proportions < min_proportion]

    assert violating_groups.empty, (
        f"Grupo(s) de '{protected_attr}' por debajo del umbral de "
        f"representatividad ({min_proportion}):\n{violating_groups}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
