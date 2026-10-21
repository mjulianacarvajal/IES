import joblib
import numpy as np
import pandas as pd

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    Ridge
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from src.config import (
    TABLES_DIR,
    MODELS_DIR,
)


def compare_models(df):

    target = "Costo_Total"

    num_cols = [
        c for c in [
            "Duration_Years",
            "Tuition_USD",
            "Living_Cost_Index",
            "Rent_USD",
            "Visa_Fee_USD",
            "Insurance_USD",
            "Exchange_Rate",
        ]
        if c in df.columns
    ]

    cat_cols = [
        c for c in [
            "Country",
            "City",
            "University",
            "Program",
            "Level",
        ]
        if c in df.columns
    ]

    X = df[
        num_cols +
        cat_cols
    ]

    y = df[target]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    preprocessor = (
        ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    num_cols,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown=
                        "ignore"
                    ),
                    cat_cols,
                ),
            ]
        )
    )

    models = {
        "Ridge":
            Ridge(alpha=1.0),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
            ),

        "GradientBoosting":
            GradientBoostingRegressor(
                random_state=42
            ),
    }

    results = []

    best_pipeline = None
    best_r2 = -999

    for name, estimator in models.items():

        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ])

        pipeline.fit(
            X_train,
            y_train,
        )

        pred = pipeline.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred,
            )
        )

        r2 = r2_score(
            y_test,
            pred,
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipeline

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        TABLES_DIR /
        "model_comparison.csv",
        index=False,
    )

    joblib.dump(
        best_pipeline,
        MODELS_DIR /
        "best_model.pkl",
    )

    return (
        result_df
        .sort_values(
            "R2",
            ascending=False
        )
        .iloc[0]
        .to_dict()
    )