"""
exploratory_analysis.py

Análisis exploratorio de datos (EDA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.config import (
    TABLES_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
)


# ==========================================================
# CALIDAD DE DATOS
# ==========================================================

def data_quality(df):

    report = []

    report.append("# Calidad de Datos\n")

    report.append(
        f"Registros: {df.shape[0]}"
    )

    report.append(
        f"Columnas: {df.shape[1]}"
    )

    report.append("\n## Valores Nulos\n")

    report.append(
        df.isnull().sum().to_string()
    )

    report.append("\n## Registros Duplicados\n")

    report.append(
        str(df.duplicated().sum())
    )

    with open(
        REPORTS_DIR / "data_quality.md",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(report)
        )


# ==========================================================
# ESTADISTICAS DESCRIPTIVAS
# ==========================================================

def descriptive_statistics(df):

    stats = (
        df.describe(include="all")
        .transpose()
    )

    stats.to_csv(
        TABLES_DIR /
        "descriptive_statistics.csv"
    )

    return stats


def _correlation_matrix(df):

    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr()

    if correlation.empty:
        raise ValueError(
            "No hay columnas numéricas para generar la matriz de correlación."
        )

    return correlation


def _average_cost_by_country_program_level(df):

    group_columns = [
        "Country",
        "Program",
        "Level",
    ]
    missing_columns = [
        column
        for column in [*group_columns, "Costo_Total"]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas para calcular el costo promedio: {missing_columns}"
        )

    return (
        df.groupby(group_columns)["Costo_Total"]
        .mean()
        .reset_index()
        .sort_values(
            by=["Country", "Costo_Total"],
            ascending=[True, False],
        )
    )


# ==========================================================
# TABLAS AGREGADAS
# ==========================================================

def generate_tables(df):

    _correlation_matrix(df).to_csv(
        TABLES_DIR / "correlation_matrix.csv"
    )

    _average_cost_by_country_program_level(df).to_csv(
        TABLES_DIR / "average_cost_by_country_program_level.csv",
        index=False,
    )

    if "Country" in df.columns:

        country_costs = (
            df.groupby("Country")
            ["Costo_Total"]
            .mean()
            .sort_values(ascending=False)
        )

        country_costs.to_csv(
            TABLES_DIR /
            "country_costs.csv"
        )

    if "Program" in df.columns:

        program_costs = (
            df.groupby("Program")
            ["Costo_Total"]
            .mean()
            .sort_values(ascending=False)
        )

        program_costs.to_csv(
            TABLES_DIR /
            "program_costs.csv"
        )

    if "Level" in df.columns:

        level_costs = (
            df.groupby("Level")
            ["Costo_Total"]
            .mean()
            .sort_values(ascending=False)
        )

        level_costs.to_csv(
            TABLES_DIR /
            "level_costs.csv"
        )


# ==========================================================
# EXPORTACION EXCEL
# ==========================================================

def generate_excel_report(df):

    excel_file = (
        TABLES_DIR /
        "results.xlsx"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl"
    ) as writer:

        (
            df.describe(include="all")
            .transpose()
            .to_excel(
                writer,
                sheet_name="Statistics"
            )
        )

        _correlation_matrix(df).to_excel(
            writer,
            sheet_name="Correlation",
        )

        _average_cost_by_country_program_level(df).to_excel(
            writer,
            sheet_name="Country_Program_Level",
            index=False,
        )

        if "Country" in df.columns:

            (
                df.groupby("Country")
                ["Costo_Total"]
                .mean()
                .sort_values(
                    ascending=False
                )
                .to_excel(
                    writer,
                    sheet_name="Countries"
                )
            )

        if "Program" in df.columns:

            (
                df.groupby("Program")
                ["Costo_Total"]
                .mean()
                .sort_values(
                    ascending=False
                )
                .to_excel(
                    writer,
                    sheet_name="Programs"
                )
            )

        if "Level" in df.columns:

            (
                df.groupby("Level")
                ["Costo_Total"]
                .mean()
                .sort_values(
                    ascending=False
                )
                .to_excel(
                    writer,
                    sheet_name="Levels"
                )
            )


# ==========================================================
# MATRIZ DE CORRELACION
# =======================================

def generate_plots(df):

    if "Costo_Total" not in df.columns:
        raise ValueError(
            "La columna 'Costo_Total' es necesaria para generar sus gráficos."
        )

    costs = pd.to_numeric(
        df["Costo_Total"],
        errors="coerce",
    ).dropna()

    if costs.empty:
        raise ValueError(
            "No hay valores válidos en la columna 'Costo_Total' para graficar."
        )

    correlation = _correlation_matrix(df)

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Matriz de correlación")

    fig.colorbar(
        image,
        ax=ax,
        label="Correlación",
    )
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "correlation_matrix.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    _, bin_edges, _ = ax.hist(
        costs,
        bins=30,
        color="steelblue",
        edgecolor="white",
    )

    if len(costs) > 1 and costs.nunique() > 1:
        x_values = np.linspace(costs.min(), costs.max(), 200)
        kde = gaussian_kde(costs)
        bin_width = bin_edges[1] - bin_edges[0]
        ax.plot(
            x_values,
            kde(x_values) * len(costs) * bin_width,
            color="darkred",
            linewidth=2,
            label="KDE",
        )
        ax.legend()

    ax.set_title("Distribución del costo total")
    ax.set_xlabel("Costo total (USD)")
    ax.set_ylabel("Frecuencia")
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "costo_total_distribution.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )
    ax.boxplot(costs)
    ax.set_title("Diagrama de caja del costo total")
    ax.set_ylabel("Costo total (USD)")
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.7,
    )
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "costo_total_boxplot.png",
        bbox_inches="tight",
    )
    plt.close(fig)