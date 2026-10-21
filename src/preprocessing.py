import numpy as np


def validate_columns(df):

    required = [
        "Tuition_USD",
        "Duration_Years",
        "Rent_USD",
        "Insurance_USD",
        "Visa_Fee_USD",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}"
        )


def clean_data(df):

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    object_cols = df.select_dtypes(
        exclude=np.number
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    for col in object_cols:
        df[col] = df[col].fillna(
            "Desconocido"
        )

    return df


def create_total_cost(df):

    df["Costo_Total"] = (
        df["Tuition_USD"]
        * df["Duration_Years"]
        + df["Rent_USD"]
        * 12
        * df["Duration_Years"]
        + df["Insurance_USD"]
        * df["Duration_Years"]
        + df["Visa_Fee_USD"]
    )

    return df