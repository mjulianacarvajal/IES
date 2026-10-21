from src.config import (
    REPORTS_DIR
)


def executive_report(metrics):

    report = f"""
# Executive Report

## Mejor Modelo

Modelo: {metrics['Model']}

MAE: {metrics['MAE']:.2f}

RMSE: {metrics['RMSE']:.2f}

R²: {metrics['R2']:.4f}

## Conclusiones

1. El costo de matrícula es una variable crítica.

2. El costo de renta impacta
fuertemente el costo total.

3. Los modelos de ML logran
alta capacidad predictiva.
"""

    with open(
        REPORTS_DIR /
        "executive_report.md",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)


def generate_profile_report(df):

    try:

        from ydata_profiling import (
            ProfileReport
        )

        profile = ProfileReport(
            df,
            title=(
                "Education Costs"
            ),
        )

        profile.to_file(
            REPORTS_DIR /
            "profile.html"
        )

    except Exception:
        pass