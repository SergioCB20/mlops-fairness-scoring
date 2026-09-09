import pandas as pd
import hashlib
from pathlib import Path

RAW_DATA_PATH = Path("zona1_ingesta/data/raw/credit-scoring.csv")

EXPECTED_COLUMNS = {
    "label", "Age", "Sex", "Marital", "Region",
    "Number_of_credits", "Linked_cards"
}

LEAKAGE_COLUMNS = ["Score_level", "Score_class", "Score_point"]

def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

def compute_dvc_hash(filepath: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_raw_batch(filepath: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, str]:
    df = drop_leakage_columns(pd.read_csv(filepath))
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Columnas esperadas ausentes: {missing}")
    batch_hash = compute_dvc_hash(filepath)
    print(f"Batch cargado: {len(df)} registros | hash: {batch_hash}")
    return df, batch_hash

if __name__ == "__main__":
    df, batch_hash = load_raw_batch()
    print(df.head())
    print(f"\nDistribución de label:\n{df['label'].value_counts()}")
