import pandas as pd

from src.config import DATA_FILE


def load_data():

    df = pd.read_csv(DATA_FILE)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df