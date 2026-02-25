"""Utilitaires d'évaluation des modèles de regression.

Ce module centralise le calcul des métriques, l'affichage des performances
et des visualisations de diagnostic (résidus, importances de variables).
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from energy_sights.config import FIGURES_DIR
from energy_sights.logging_config import setup_logging, get_task_logger

setup_logging()

log = get_task_logger(task_name='model_evaluation')

from typing import Dict, Any, Optional
from numpy.typing import NDArray

def evaluate_regression(y_true: NDArray | pd.DataFrame, y_pred: NDArray | pd.DataFrame, model_name: str='') -> Dict[str, Any]:
    """Calcule les métriques principales d'un modèle de regression.

    Args:
        y_true: Valeurs réelles de reference.
        y_pred: Valeurs prédites par le modèle.
        model_name: Nom du modèle évalué.

    Returns:
        Un dictionnaire contenant `r2_score`, `RMSE`, `MAE`, `MAPE`
        ainsi que `model_name`.
    """
    # this method will compute the model error
    r2: float = r2_score(y_true=y_true, y_pred=y_pred)

    rmse: float = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

    mae: float = mean_absolute_error(y_true=y_true, y_pred=y_pred)

    mape: float = mean_absolute_percentage_error(y_pred=y_pred, y_true=y_true)

    metrics: Dict[str, float] = {
        'model_name': model_name,
        'r2_score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

    return metrics

# Plots setup
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
sns.set_style(style='darkgrid')


def plot_residuals(y_true: NDArray, y_pred: NDArray, save_plot: Optional[str]=None, model_name: str="") -> None:
    """Trace les diagnostics de prediction et de résidus.

    Le graphique produit deux vues:
    - valeurs réelles vs predictions
    - distribution des résidus

    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        save_plot: Nom de fichier de sortie optionnel.
        model_name: Libelle du modèle affiche sur le graphe.
    """

    # Fixing the dimension of the arrays
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    sns.scatterplot(x=y_true,
                    y=y_pred,
                    alpha=.6,
                    color='#E4FDE1',
                    lw=2.2, ec='k',
                    s=60, ax=axes[0])
    axes[0].set_xlabel(xlabel='Real Values')
    axes[0].set_ylabel(ylabel='Predicted Values')
    axes[0].set_title(label=f'{model_name} predictions vs reality')

    # The 2nd graph
    residuals = y_true - y_pred
    sns.histplot(data=None, x=residuals, kde=True,
                 element='bars', color='#DADFF7', lw=2.2, ec='k')
    axes[1].set_xlabel(xlabel='Errors (Predictions vs Reality)')
    axes[1].set_title(label='Errors Distribution')
    # let's also add a trend line
    axes[1].axvline(x=0, color='#A41623', linestyle='--')

    plt.tight_layout()

    if isinstance(save_plot, str):
        if not save_plot.endswith('.png'):
            save_plot += '.png'
            log.info('We automatically add the extension .png')
        save_path: Path = FIGURES_DIR / save_plot

        fig.savefig(fname=save_path, dpi=300, bbox_inches='tight')
        log.info(f"The residuals plot of the {model_name} successfully saved at: {save_path}")
    plt.show()


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Affiche les métriques de regression dans un format lisible.

    Args:
        metrics: Dictionnaire produit par `evaluate_regression`.
    """
    print(f"\n Model performance : {metrics['model_name']}")
    print(f"  RMSE (Root Mean Squared Error) : {metrics['RMSE']:,.2f}")
    print(f"  MAE  (Mean Absolute Error)     : {metrics['MAE']:,.2f}")
    print(f"  MAPE (Error %)           : {metrics['MAPE']:.2%}")  # Percentage format
    print(f"  R²   (Score explication)  : {metrics['r2_score']:.4f}")


def plot_features_importance(model, feature_names: list[str], save_plot: Optional[str]=None) -> None:
    """Visualise l'importance des variables pour les modèles compatibles.

    Args:
        model: Estimateur ou `TransformedTargetRegressor`.
        feature_names: Liste ordonnée des noms de variables.
        save_plot: Nom de fichier de sortie optionnel.
    """

    if isinstance(model, TransformedTargetRegressor):
        estimator = model.regressor_.named_steps['model']

    else:
        estimator = model

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_

    else:
        print("This model don't provide importance features.")
        return

    features_imp: pd.DataFrame = pd.DataFrame(data={
        'Value': importances,
        'Features': feature_names
    })

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    sns.barplot(data=features_imp.sort_values(by='Value', ascending=False),
                x='Value',
                y='Features',
                color='#FCF7F8',
                lw=2.2,
                ec='k',
                ax=ax)

    ax.set_title(label="What's the most important features ?")
    plt.tight_layout()
    if isinstance(save_plot, str):
        if not save_plot.endswith('png'):
            save_plot += '.png'
            log.info(f'The extension of the file is not correct ! We add the .png automatically.')

        save_path: Path = FIGURES_DIR / save_plot

        fig.savefig(fname=save_path, dpi=300, bbox_inches='tight')

        log.info(f'Graphic saved successfully at: {save_path}')

    plt.show()


def print_feature_importances(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Retourne les importances de variables sous forme tabulaire.

    Args:
        model: Estimateur ou `TransformedTargetRegressor`.
        feature_names: Liste ordonnée des noms de variables.

    Returns:
        Un DataFrame trie par importance décroissante, ou `None` si le
        modèle n'expose pas `feature_importances_`.
    """
    if isinstance(model, TransformedTargetRegressor):
        estimator = model.regressor_.named_steps['model']

    else:
        estimator = model

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_

    else:
        print(f"The model don't provide importance features")
        return None

    fea_df: pd.DataFrame = pd.DataFrame(data={
        'Features': feature_names,
        'Values': importances
    })

    return fea_df.sort_values(by='Values', ascending=False)

