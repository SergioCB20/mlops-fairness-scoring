from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ["Age", "Number_of_credits", "Linked_cards", "INPS_mln_sum"]
CATEGORICAL_FEATURES = ["Region"]
# Nota: Sex se EXCLUYE deliberadamente de las features de entrada -
# es el atributo protegido Z, no debe ser un input directo del modelo.
# Marital también se EXCLUYE: el QA Gate 2 (Contrato C2) detectó que es
# en la práctica un proxy casi perfecto de Sex (6 de 7 categorías con
# separación 100% en el crosstab Marital x Sex sobre 8,707 registros) -
# evidencia documentada de leakage indirecto, no una decisión arbitraria.

def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=True,
    )
