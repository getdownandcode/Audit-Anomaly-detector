"""Download and load the public ULB credit-card transactions dataset."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "creditcard.csv"
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def load_data() -> pd.DataFrame:
    """Load the cached Kaggle CSV, downloading it once when necessary."""
    if not DATA_PATH.exists():
        try:
            import kagglehub

            source = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud")) / "creditcard.csv"
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            DATA_PATH.write_bytes(source.read_bytes())
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the Kaggle dataset ({exc}). Set up Kaggle credentials or "
                f"download {DATASET_URL} manually to {DATA_PATH}."
            ) from exc
    return pd.read_csv(DATA_PATH)
