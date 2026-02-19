"""Configuration centralisée du logging applicatif.

Ce module fournit:
- une configuration globale des handlers Loguru
- un logger specialise par tache avec sortie fichier dédiée.
"""

from loguru import logger
from pathlib import Path
import sys

# Creation of the logging directory
logs_dir: Path = Path(__file__).resolve().parents[1] / 'logs'
logs_dir.mkdir(exist_ok=True)

def setup_logging():
    """Initialise les handlers Loguru de base.

    La configuration actuelle redirige les logs de niveau INFO vers la sortie
    standard avec un format enrichi.
    """
    # removal of the old handler
    logger.remove()

    # Building a new handler file

    logger.add(
        sys.stdout,
        level='INFO',
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
    )

    # adding of an optional handler, the console
    logger.add(
        sink=sys.stdout, level='INFO'
    )


def get_task_logger(task_name: str):
    """Retourne un logger lie a une tache et route vers un fichier dedie.

    Args:
        task_name: Nom fonctionnel de la tache (ex: `preprocessing`).

    Returns:
        Un logger Loguru avec contexte `task` et filtre d'écriture associe.
    """

    log_file = logs_dir / f"{task_name}.log"

    # let's define a filter to ensure each file have its own logs file
    # We only want the files with the attribute task equal the task name
    task_filter = lambda record: record['extra'].get('task') == task_name

    # let's add this file as the sink (destination) of the logs
    logger.add(
        sink=log_file,
        filter=task_filter,
        level='INFO',
        rotation="10 MB"    # This creates a new file when the file exceed 10 MB
    )

    # now we return the logger with the task label
    return logger.bind(task=task_name)
