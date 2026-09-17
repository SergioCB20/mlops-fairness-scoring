"""
Entrena el clasificador de scoring sobre el pipeline de Zona 1
(ya limpio de Sex y Marital tras la corrección del QA Gate 2).
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zona1_ingesta.ingestor import load_raw_batch, drop_leakage_columns
from zona1_ingesta.transformers.pipeline import build_preprocessing_pipeline

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_model_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(
            class_weight="balanced",  # compensa el desbalance 92%/8% del label
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def load_train_test_split():
    df, _ = load_raw_batch()
    df = drop_leakage_columns(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    return train_df, test_df


def train_and_predict():
    train_df, test_df = load_train_test_split()

    model = build_model_pipeline()
    model.fit(train_df, train_df["label"])

    y_pred = model.predict(test_df)
    return {
        "model": model,
        "y_true": test_df["label"].values,
        "y_pred": y_pred,
        "z_test": test_df["Sex"].values,
    }


if __name__ == "__main__":
    result = train_and_predict()
    print(f"Predicciones generadas sobre {len(result['y_true'])} registros de prueba.")
