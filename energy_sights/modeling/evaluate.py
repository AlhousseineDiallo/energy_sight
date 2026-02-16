import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from typing import Dict, Any
from numpy.typing import NDArray

def evaluate_regression(y_true: NDArray | pd.DataFrame, y_pred: NDArray | pd.DataFrame, model_name: str='') -> Dict[str, Any]:
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


def plot_residuals(y_true: NDArray, y_pred: NDArray, model_name: str="") -> None:

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.squeeze()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))
    sns.scatterplot(x=y_true,
                    y=y_pred,
                    alpha=.6,
                    color='#878472',
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
    plt.show()


def print_metrics(metrics: Dict[str, Any]) -> None:
    print(f"\n Model performance : {metrics['model_name']}")
    print(f"  RMSE (Root Mean Squared Error) : {metrics['RMSE']:,.2f}")
    print(f"  MAE  (Mean Absolute Error)     : {metrics['MAE']:,.2f}")
    print(f"  MAPE (Error %)           : {metrics['MAPE']:.2%}")  # Percentage format
    print(f"  R²   (Score explication)  : {metrics['r2_score']:.4f}")


def plot_features_importance(model, feature_names: list[str]) -> None:

    if isinstance(model, Pipeline):
        estimator = model.named_steps['model']

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
                hash='/',
                color='#FCF7F8',
                lw=2.2,
                ec='k',
                ax=ax)

    ax.set_title(label="What's the most important features ?")
    plt.tight_layout()
    plt.show()
