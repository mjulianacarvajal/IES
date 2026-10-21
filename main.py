
from src.utils import (
    create_directories
)

from src.pipeline import (
    run_pipeline
)


def main():

    create_directories()

    run_pipeline()


if __name__ == "__main__":
    main()