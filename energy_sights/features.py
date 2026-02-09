import pandas as pd
import numpy as np
from energy_sights.logging_config import setup_logging, get_task_logger
from energy_sights.config import center_lat, center_long

setup_logging()

# Recovery of the logs in the proper file
log = get_task_logger(task_name='feature_engineering')

def create_binarization(df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = df.copy()
    log.info('Binarization of the features')
    df['IsComplex'] = (df['NumberofBuildings'] > 1).astype(dtype=pd.Int32Dtype())

    # now we can remove the NumberofBuilding column
    df = df.drop(columns=['NumberofBuildings'])

    return df


def create_date(df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = df.copy()
    data_year: int = 2016
    log.info('Creation of the column based on the BuiltYear.')
    df['BuildingAge'] = data_year - df['YearBuilt']
    return df


def haversine(lat1: float, long1: float, lat2: float, long2: float) -> float:
    earth_radius: int = 6370 # Km
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
    df: pd.DataFrame = df.copy()

    df['AverageFloorArea'] = df['PropertyGFATotal'] / df['NumberofFloors']

    df['DistToCenter'] = df.apply(func=lambda row: haversine(row['Latitude'], center_lat, row['Longitude'], center_long),
                                  axis=1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info('Launching the features creation pipeline')

    processed_df: pd.DataFrame = df.pipe(func=create_binarization)\
                                    .pipe(func=create_date)\
                                    .pipe(func=create_characteristic)

    log.sucess(f"New columns: {len(processed_df.columns) - len(df.columns)}")

    return processed_df
