"""Utilitaires de visualisation pour l'analyse exploratoire.

Le module propose des graphiques de distribution (statiques et interactifs)
et des matrices de correlation.
"""

from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from energy_sights.logging_config import setup_logging, get_task_logger
from energy_sights.config import FIGURES_DIR

setup_logging()

log = get_task_logger(task_name='plots')


def plot_distribution(df: pd.DataFrame, col: str, save_plot: Optional[str]=None, bins: int=10) -> None:
    """Trace la distribution d'une variable numérique (boxplot + histogramme).

    Args:
        df: DataFrame source.
        col: Colonne à visualiser.
        save_plot: Nom de fichier de sauvegarde optionnel.
        bins: Nombre de classes de l'histogramme.
    """
    log.info(f"Plotting distribution for column: {col}")
    fig, axes = plt.subplots(nrows=2, ncols=1, layout='tight', figsize=(14, 7))
    sns.boxplot(data=df,
                x=col,
                whis=1.5,
                linecolor='k',
                linewidth=2.2,
                ax=axes[0],
                color='#F4F4F9')
    fig.suptitle(t=f"Distribution of {col}", fontsize=14, fontweight='bold')

    sns.histplot(data=df,
                 x=col,
                 kde=True,
                 element='bars',
                 lw=2.2,
                 edgecolor='k',
                 stat='count',
                 ax=axes[1],
                 color='#586F7C',
                 bins=bins)
    axes[1].set_ylabel(ylabel='')

    plt.tight_layout()

    if isinstance(save_plot, str):
        if not save_plot.endswith('.png'):
            # suppression des espaces pour avoir des noms conformes aux conventions
            save_plot: str = re.sub(pattern=r'\s+', repl='_', string=save_plot.strip()).lower()
            save_plot += '.png'
            log.info('We automatically add the extension .png')

        save_path: Path = FIGURES_DIR / save_plot
        fig.savefig(fname=save_path, dpi=300, bbox_inches='tight')
        log.info(f'Graphic: {col} distribution saved successfully at: {save_path}.')

    plt.show()
    log.success(f'Plot generation for the column {col} complete')


def interactive_distribution(df: pd.DataFrame, col: str, save_plot: Optional[str]=None, n_bins: int=50) -> None:
    """Trace une distribution interactive avec Plotly.

    Args:
        df: DataFrame source.
        col: Colonne numérique à visualiser.
        save_plot: Nom de fichier de sauvegarde optionnel.
        n_bins: Nombre de classes de l'histogramme.

    Raises:
        TypeError: Si la colonne cible n'est pas numérique.
    """
    if not is_numeric_dtype(df[col]):
        log.error(f'The column {col} must be numeric.')
        raise TypeError(f"The column {col} must be numeric")
    fig = px.histogram(data_frame=df,
                       x=col,
                       opacity=.7,
                       marginal='box',
                       title=f"{col} Distribution",
                       nbins=n_bins)
    fig.update_layout(bargap=.1)
    if isinstance(save_plot, str):
        if not save_plot.endswith('.png'):
            # Procédons à l'usage de regex pour assurer une meilleure qualité du code.
            save_plot: str = re.sub(pattern=r'\s+', repl='_', string=save_plot.strip()).lower()
            save_plot += '.png'
            log.info('The name is not correct ! We add the extension .png automatically.')

        save_path: Path = FIGURES_DIR / save_plot
        fig.write_image(file=save_path, engine='kaleido')
        log.info(f'Graphic: {col} interactive distribution saved successfully at: {save_path}')
    fig.show()


def heat_correlation(df: pd.DataFrame, save_plot: Optional[str]=None, num_only: bool=True, method: str='pearson') -> None:
    """Affiche une matrice de correlation sous forme de heatmap.

    Args:
        df: DataFrame source.
        save_plot: Nom de fichier de sauvegarde optionnel.
        num_only: Si vrai, limite le calcul aux colonnes numériques.
        method: Methode de correlation (`pearson`, `spearman`, `kendall`).

    Raises:
        ValueError: Si la methode n'est pas supportée.
    """
    possible_methods: list[str] = ['pearson', 'spearman', 'kendall']
    if df.empty:
        log.error("heat_correlation is impossible: the dataframe is empty.")
        return

    if num_only:
        df_numeric = df.select_dtypes(include='number')
        if len(df_numeric) == 0:
            log.warning(f"heat_correlation: any numeric fields with (numeric_only=True).")
            return

    if method not in possible_methods:
        log.error(f"Your method don't exist, it must be one of those: {possible_methods}.")
        raise ValueError(f"This method {method} is not available ! It must be one of those: {possible_methods}")

    fig, ax = plt.subplots(nrows=1, ncols=1)
    data_correlation: pd.DataFrame = df.corr(method=method, numeric_only=num_only)
    sns.heatmap(data=data_correlation, cmap='Blues', linewidths=2.2, linecolor='white', annot=True, fmt='.3f', ax=ax)

    fig.suptitle(t='Correlation in the Data')
    plt.tight_layout()

    if isinstance(save_plot, str):
        if not save_plot.endswith('.png'):
            save_plot: str = re.sub(pattern=r'\s+', repl='_', string=save_plot.strip()).lower()
            save_plot += '.png'
            log.info('We automatically add the extension .png')
        save_path: Path = FIGURES_DIR / save_plot

        fig.savefig(fname=save_path, dpi=300, bbox_inches='tight')

        log.info(f"The correlation matrix has been successfully saved at: {save_path}")

    plt.show()

    log.info(
        f'The heat_correlation is done: corr_shape {data_correlation.shape}'
    )
