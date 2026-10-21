import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt

from src.config import (
    MODELS_DIR,
    TABLES_DIR,
    PLOTS_DIR,
)


def train_neural_network(df):

    target = "Costo_Total"

    numeric_features = [
        c
        for c in [
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

    categorical_features = [
        c
        for c in [
            "Country",
            "City",
            "University",
            "Program",
            "Level",
        ]
        if c in df.columns
    ]

    X = df[
        numeric_features +
        categorical_features
    ]

    y = df[target]

    preprocessor = (
        ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numeric_features,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_features,
                ),
            ]
        )
    )

    X_processed = preprocessor.fit_transform(X)

    if hasattr(
        X_processed,
        "toarray"
    ):
        X_processed = (
            X_processed.toarray()
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_processed,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            64,
            activation="relu",
            input_shape=(
                X_train.shape[1],
            )
        ),
        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),
        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[
            early_stop
        ],
        verbose=1,
    )

    predictions = (
        model.predict(X_test)
        .flatten()
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    metrics_df = pd.DataFrame(
        [{
            "Model":
                "NeuralNetwork",
            "MAE":
                mae,
            "RMSE":
                rmse,
            "R2":
                r2,
        }]
    )

    metrics_df.to_csv(
        TABLES_DIR /
        "neural_network_metrics.csv",
        index=False,
    )

    model.save(
        MODELS_DIR /
        "neural_network.keras"
    )

    plt.figure(
        figsize=(10, 5.5)
    )

    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Model Loss Over Epochs"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss (Mean Squared Error)")

    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "training_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "Model":
            "NeuralNetwork",
        "MAE":
            mae,
        "RMSE":
            rmse,
        "R2":
            r2,
    }