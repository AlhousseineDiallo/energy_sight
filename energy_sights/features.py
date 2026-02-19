"""Fonctions d'ingénierie de variables.

Ce module construit des variables dérivées utiles a la modelisation de la
consommation énergétique et des emissions.
"""

import pandas as pd
import numpy as np
from energy_sights.logging_config import setup_logging, get_task_logger
from energy_sights.config import center_lat, center_long

setup_logging()

# Recovery of the logs in the proper file
log = get_task_logger(task_name='feature_engineering')

def create_binarization(df: pd.DataFrame) -> pd.DataFrame:
    """Cree une variable binaire de complexité alimentaire.

    Règle appliquée:
    - `IsComplex = 1` si `NumberofBuildings > 1`
    - `IsComplex = 0` sinon

    La colonne source `NumberofBuildings` est ensuite retiree.

    Args:
        df: DataFrame source.

    Returns:
        DataFrame enrichi avec `IsComplex`.
    """
    df: pd.DataFrame = df.copy()
    log.info('Binarization of the features')
    df['IsComplex'] = (df['NumberofBuildings'] > 1).astype(dtype=pd.Int32Dtype())

    # now we can remove the NumberofBuilding column
    df = df.drop(columns=['NumberofBuildings'])

    return df


def create_date(df: pd.DataFrame) -> pd.DataFrame:
    """Cree la variable d'age du bâtiment.

    L'age est calcule par difference entre l'année de reference (2016)
    et l'année de construction.

    Args:
        df: DataFrame source.

    Returns:
        DataFrame avec la colonne `BuildingAge`.
    """
    df: pd.DataFrame = df.copy()
    data_year: int = 2016
    log.info('Creation of the column based on the BuiltYear.')
    df['BuildingAge'] = data_year - df['YearBuilt']
    return df


def haversine(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calcule la distance géodésique (Haversine) entre deux points.

    Args:
        lat1: Latitude du premier point.
        long1: Longitude du premier point.
        lat2: Latitude du second point.
        long2: Longitude du second point.

    Returns:
        Distance en kilometres.
    """
    earth_radius: int = 6371 # Km
    try:
        # conversion in radians
        lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
        delta_lat: float = np.radians(lat2 - lat1)
        delta_long: float = np.radians(long2 - long1)
        a: float = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) *  np.cos(lat2_rad) * np.sin(delta_long / 2) ** 2

        c: float = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return earth_radius * c

    except (ValueError, TypeError) as e:
        log.error(f"An error occurs in the haversine function: {e}")


def create_characteristic(df: pd.DataFrame) -> pd.DataFrame:
    """Cree des variables structurelles et spatiales dérivées.

    Variables produites:
    - `AverageFloorArea`
    - `DistToCenter`
    - `ParkingRatio`

    Args:
        df: DataFrame source.

    Returns:
        DataFrame enrichi.
    """
    df: pd.DataFrame = df.copy()

    df['AverageFloorArea']: pd.Series = df['PropertyGFATotal'] / df['NumberofFloors']

    df['DistToCenter']: pd.Series = df.apply(func=lambda row: haversine(row['Latitude'], center_lat, row['Longitude'], center_long),
                                  axis=1)

    df['ParkingRatio']: pd.Series = df['PropertyGFAParking'] / df['PropertyGFATotal']

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execute le pipeline complet de feature engineering.

    Args:
        df: DataFrame source.

    Returns:
        DataFrame transforme apres enchainement des étapes de creation
        de variables.
    """
    log.info('Launching the features creation pipeline')

    processed_df: pd.DataFrame = df.pipe(func=create_binarization)\
                                    .pipe(func=create_date)\
                                    .pipe(func=create_characteristic)

    log.success(f"New columns: {len(processed_df.columns) - len(df.columns)}")

    return processed_df
