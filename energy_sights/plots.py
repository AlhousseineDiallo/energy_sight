import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from energy_sights.logging_config import setup_logging, get_task_logger

setup_logging()

logger = get_task_logger(task_name='plots')


def plot_distribution(df: pd.DataFrame, col: str, bins: int=10) -> None:
    """
    Plot the distribution of a numerical variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Numerical column to plot
    bins: int
        Numerical the number of bins
    """
    logger.info(f"Plotting distribution for column: {col}")
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
    plt.show()
    logger.success(f'Plot generation for the column {col} complete')


def interactive_distribution(df: pd.DataFrame, col: str, n_bins: int=50) -> None:
    if not is_numeric_dtype(df[col]):
        logger.error(f'The column {col} must be numeric.')
        raise TypeError(f"The column {col} must be numeric")
    fig = px.histogram(data_frame=df,
                       x=col,
                       opacity=.7,
                       marginal='box',
                       title=f"{col} Distribution",
                       nbins=n_bins)
    fig.update_layout(bargap=.1)
    fig.show()

