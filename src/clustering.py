import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.config import (
    TABLES_DIR
)


def clustering_analysis(df):

    features = [
        "Tuition_USD",
        "Rent_USD",
        "Duration_Years",
        "Costo_Total",
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(
        df[features]
    )

    results = []

    for k in [3, 5, 7]:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
        )

        results.append({
            "Clusters": k,
            "Silhouette": score,
        })

    pd.DataFrame(
        results
    ).to_csv(
        TABLES_DIR /
        "cluster_summary.csv",
        index=False,
    )