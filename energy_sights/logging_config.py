from loguru import logger
from pathlib import Path
import sys

def setup_logging(log_file: Path):
    # creation of the logging folder
    log_file.parents[0].mkdir(parents=True, exist_ok=True)

    # removal of the old handler
    logger.remove()

    # Building a new handler file

    logger.add(
        sink=log_file,
        level='INFO',
        format="{time} {level} {message}"
    )

    # adding of an optional handler, the console
    logger.add(
        sink=sys.stdout, level='INFO'
    )