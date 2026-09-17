"""
QA Gate 2: No-Leakage Indirecto vía Información Mutua Condicional (Contrato C2)
I(X_trans; Z | Y) <= umbral, estratificado por cada valor de Y.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.feature_selection import mutual_info_classif

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zona1_ingesta.ingestor import load_raw_batch, drop_leakage_columns
from zona1_ingesta.transformers.pipeline import build_preprocessing_pipeline

CONTRACT_PATH = Path("contracts/c2_leakage_indirecto.yaml")


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def transformed_data(contract):
    df, _ = load_raw_batch()
    df = drop_leakage_columns(df)

    protected_attr = contract["conditional_mutual_information"]["protected_attribute"]
    target_attr = contract["conditional_mutual_information"]["target_attribute"]

    pipeline = build_preprocessing_pipeline()
    X_trans = pipeline.fit_transform(df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    return {
        "X_trans": X_trans,
        "feature_names": pipeline.get_feature_names_out(),
        "Z": df[protected_attr].values,
        "Y": df[target_attr].values,
    }


def compute_conditional_mi(X_trans, Z, Y, n_neighbors, random_state) -> float:
    """
    Calcula I(X_trans; Z | Y) estratificando por cada valor de Y:
    promedia la MI(X_trans, Z) calculada dentro de cada subconjunto Y=y,
    ponderada por el tamaño de cada subconjunto.
    """
    total_mi = 0.0
    n_total = len(Y)

    for y_value in np.unique(Y):
        mask = Y == y_value
        X_subset = X_trans[mask]
        Z_subset = Z[mask]

        if len(np.unique(Z_subset)) < 2 or len(X_subset) < n_neighbors + 1:
            continue  # subgrupo insuficiente para estimar MI de forma confiable

        mi_per_feature = mutual_info_classif(
            X_subset, Z_subset,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        weight = mask.sum() / n_total
        total_mi += weight * mi_per_feature.max()

    return total_mi


def test_conditional_mutual_information(transformed_data, contract):
    """
    Verifica que ninguna feature transformada exceda el umbral de
    información mutua condicional respecto al atributo protegido.
    """
    mi_config = contract["conditional_mutual_information"]
    threshold = mi_config["max_mi_threshold"]

    cmi = compute_conditional_mi(
        transformed_data["X_trans"],
        transformed_data["Z"],
        transformed_data["Y"],
        n_neighbors=mi_config["n_neighbors"],
        random_state=mi_config["random_state"],
    )

    assert cmi <= threshold, (
        f"I(X_trans; Z | Y) = {cmi:.4f} excede el umbral del Contrato C2 "
        f"({threshold}). El pipeline de transformación introduce leakage "
        f"indirecto del atributo protegido '{mi_config['protected_attribute']}'."
    )
    print(f"\n✅ I(X_trans; Z | Y) = {cmi:.4f} (umbral: {threshold})")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
