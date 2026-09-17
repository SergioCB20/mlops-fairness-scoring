"""
Mitigación post-procesamiento del modelo base mediante ThresholdOptimizer
(Fairlearn), ajustando el umbral de decisión por grupo protegido para
satisfacer el Contrato C3 sin reentrenar el clasificador base.
"""
import sys
from pathlib import Path

from fairlearn.postprocessing import ThresholdOptimizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zona2_ci_training.train import build_model_pipeline, load_train_test_split, RANDOM_STATE


def train_predict_and_mitigate(constraint: str = "equalized_odds"):
    train_df, test_df = load_train_test_split()

    # 1. Entrena el modelo base (sin Sex/Marital como features, como ya validamos)
    base_model = build_model_pipeline()
    base_model.fit(train_df, train_df["label"])

    # 2. Envuelve el modelo YA ENTRENADO con el optimizador de umbral.
    #    prefit=True porque base_model ya fue entrenado en el paso anterior.
    mitigator = ThresholdOptimizer(
        estimator=base_model,
        constraints=constraint,
        predict_method="predict_proba",
        prefit=True,
    )

    # 3. El optimizador SÍ necesita el atributo protegido para calibrar
    #    los umbrales por grupo — pero solo en esta etapa de post-proceso,
    #    nunca como feature de entrada del modelo base.
    mitigator.fit(train_df, train_df["label"], sensitive_features=train_df["Sex"])

    y_pred_mitigated = mitigator.predict(
            test_df,
            sensitive_features=test_df["Sex"],
            random_state=RANDOM_STATE,
        )

    return {
        "model": mitigator,
        "y_true": test_df["label"].values,
        "y_pred": y_pred_mitigated,
        "z_test": test_df["Sex"].values,
    }


if __name__ == "__main__":
    result = train_predict_and_mitigate()
    print(f"Predicciones mitigadas generadas sobre {len(result['y_true'])} registros.")
