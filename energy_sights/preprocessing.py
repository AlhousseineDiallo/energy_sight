import pandas as pd
import numpy as np
from loguru import logger


logger.info('Starting the Preprocessing')

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_columns: list[str] = df.nunique().apply(func=lambda x: x == 1).index.tolist()
    logger.info(f"The constant columns of the dataframe are: {constant_columns}")
    df_cleaned = df.drop(columns=constant_columns)
    return df_cleaned



