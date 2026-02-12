import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from pandas.core.dtypes.common import is_numeric_dtype
from energy_sights.logging_config import setup_logging, get_task_logger

setup_logging()

log = get_task_logger(task_name='plots')


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
    plt.show()
    log.success(f'Plot generation for the column {col} complete')


def interactive_distribution(df: pd.DataFrame, col: str, n_bins: int=50) -> None:
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
    fig.show()


def heat_correlation(df: pd.DataFrame, num_only: bool=True, method: str='pearson') -> None:
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
    plt.show()

    log.info(
        f'The heat_correlation is done: corr_shape {data_correlation.shape}'
    )
