import logging

from src.config import (
    PLOTS_DIR,
    TABLES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)


def create_directories():

    folders = [
        PLOTS_DIR,
        TABLES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def create_logger():

    logger = logging.getLogger(
        "education_costs"
    )

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        handler = logging.FileHandler(
            LOGS_DIR / "analysis.log"
        )

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger