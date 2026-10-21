from src.data_loader import load_data

from src.preprocessing import (
    validate_columns,
    clean_data,
    create_total_cost,
)

from src.exploratory_analysis import (
    data_quality,
    descriptive_statistics,
    generate_tables,
    generate_excel_report,
    generate_plots,
)

from src.clustering import (
    clustering_analysis,
)

from src.modeling import (
    compare_models,
)

from src.neural_network import (
    train_neural_network,
)

from src.reporting import (
    executive_report,
    generate_profile_report,
)

from src.utils import (
    create_logger,
)


def run_pipeline():

    logger = create_logger()

    logger.info(
        "Inicio del análisis"
    )

    # ------------------------------------------------
    # 1. CARGA
    # ------------------------------------------------

    df = load_data()

    logger.info(
        f"Dataset cargado: {df.shape}"
    )

    # ------------------------------------------------
    # 2. VALIDACION
    # ------------------------------------------------

    validate_columns(df)

    logger.info(
        "Columnas requeridas validadas"
    )

    # ------------------------------------------------
    # 3. LIMPIEZA
    # ------------------------------------------------

    df = clean_data(df)

    logger.info(
        "Limpieza finalizada"
    )

    # ------------------------------------------------
    # 4. FEATURE ENGINEERING
    # ------------------------------------------------

    df = create_total_cost(df)

    logger.info(
        "Costo_Total generado"
    )

    # ------------------------------------------------
    # 5. CALIDAD
    # ------------------------------------------------

    data_quality(df)

    logger.info(
        "Reporte de calidad generado"
    )

    # ------------------------------------------------
    # 6. ESTADISTICAS
    # ------------------------------------------------

    descriptive_statistics(df)

    logger.info(
        "Estadísticas descriptivas generadas"
    )

    # ------------------------------------------------
    # 7. TABLAS
    # ------------------------------------------------

    generate_tables(df)

    logger.info(
        "Tablas agregadas generadas"
    )

    # ------------------------------------------------
    # 8. EXCEL
    # ------------------------------------------------

    generate_excel_report(df)

    logger.info(
        "Excel consolidado generado"
    )

    # ------------------------------------------------
    # 9. GRAFICOS
    # ------------------------------------------------

    generate_plots(df)

    logger.info(
        "Gráficos generados"
    )

    # ------------------------------------------------
    # 10. CLUSTERING
    # ------------------------------------------------

    clustering_analysis(df)

    logger.info(
        "Análisis KMeans completado"
    )

    # ------------------------------------------------
    # 11. MACHINE LEARNING
    # ------------------------------------------------

    metrics = compare_models(df)

    logger.info(
        f"Mejor modelo: {metrics['Model']}"
    )

    train_neural_network(df)

    logger.info(
        "Red neuronal entrenada y gráfica de pérdida generada"
    )

    # ------------------------------------------------
    # 12. PROFILING
    # ------------------------------------------------

    generate_profile_report(df)

    logger.info(
        "Profile Report generado"
    )

    # ------------------------------------------------
    # 13. REPORTE EJECUTIVO
    # ------------------------------------------------

    executive_report(metrics)

    logger.info(
        "Reporte ejecutivo generado"
    )

    logger.info(
        "Pipeline finalizado exitosamente"
    )