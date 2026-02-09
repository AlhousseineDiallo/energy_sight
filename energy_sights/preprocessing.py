import pandas as pd
import numpy as np
from pandas import Series
from pandas.api.types import is_numeric_dtype
from energy_sights.logging_config import setup_logging, get_task_logger
# let's apply the configuration of the setup_logging
setup_logging()

# Recovery of the specific logger of the preprocessing
log = get_task_logger(task_name='preprocessing')

log.info('Starting the Preprocessing')

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_columns: list[str] = df.nunique()[df.nunique().apply(func=lambda x: x == 1)].index.tolist()
    log.info(f"The constant columns of the dataframe are: {constant_columns}")
    df_cleaned = df.drop(columns=constant_columns)
    return df_cleaned


def nan_percent(df: pd.DataFrame) -> None:
    for col in df.columns:
        nulls_percents = np.mean(df[col].isna())
        print(f"{col} -> {nulls_percents: .4f} %\n")


def clean_neighborhood(df: pd.DataFrame, col: str='Neighborhood') -> pd.DataFrame:
    # entry validation
    if col not in df.columns:
        log.error(f'Column {col} not found in the Dataframe !')
        raise ValueError(f'Column {col} does not exist in {df}')

    unique_neighborhood: list[str] = df[col].unique().tolist()

    # add of a debug log
    log.debug(f"Unique values before the cleaning: {unique_neighborhood}")
    df[col]: pd.Series = df[col].str.strip().str.title()
    neighborhood_mapping: dict[str, str] = {'Delridge': 'Delridge Neighborhoods'}
    df[col] = df[col].replace(neighborhood_mapping)

    return df


def find_case_insensitive_duplicates(df: pd.DataFrame, col: str) -> dict[str, int]:
    # entry validation
    if col not in df.columns:
        log.error(f'Column {col} not found in the Dataframe !')
        raise ValueError(f'Column {col} does not exist in {df}')

    values = df[col].dropna().astype(str)
    if values.empty:
        return {}

    lower_values = values.str.lower()
    duplicates = values.groupby(by=lower_values).nunique()
    duplicated_values = duplicates[duplicates > 1].to_dict()

    if duplicated_values:
        log.info(f"Found {len(duplicated_values)} case-insensitive duplicates in '{col}'.")

    return duplicated_values


def numeric_describe(df: pd.DataFrame, col: 'str') -> Series:
    if col not in df.columns:
        log.error(f'Column {col} not found in the dataframe')
        raise ValueError(f'Column {col} not found in {df}')

    if not is_numeric_dtype(df[col]):
        log.error(f'This column is not numeric.')
        raise ValueError(f'Column must be a numeric column')

    result: pd.Series = df[col].mode()
    result.index = [f'mode_{i+1}' if len(result) > 1 else 'mode' for i in range(len(result))]
    final_description: pd.Series = pd.concat([df[col].describe(), result])
    return final_description


def logarithm(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df: pd.DataFrame = df.copy()
    if not is_numeric_dtype(df[col]):
        log.error(f"The column {col} must be numeric.")
        raise ValueError(f"Column is not numeric")

    df[f"{col}Log"]: pd.Series = np.log(df[col])
    log.info(f"The logarithm function has been used on {col} with success.")
    return df


log.success("End of the preprocessing")


