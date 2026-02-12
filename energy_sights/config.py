from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJ_ROOT / "models"

REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

center_lat: float = 47.6062
center_long: float = -122.3321
# Columns to remove (Data leakage)
Leaky_features = [
    # Energetic intensity calculated with the Site Energy Use
    'SiteEUI(kBtu/sf',
    'SiteEUIWN(kBtu/sf)',
    'SourceEUI(kBtu/sf)',
    'SourceEUIWN(kBtu/sf',
    # Intensity Calculated with the TotalGHGEmissions
    'GHGEmissionsIntensity',
    # Normalized version of the second target
    'SiteEnergyUseWN(kBtu)',
    'ENERGYSTARScore',
    'Electricity(kBtu)',
    'SteamUse(kBtu)',
    'NaturalGas(kBtu)'
]

# Useless columns for the training of the models
useless_columns = [
    'PropertyName',
    'TaxParcelIdentificationNumber',
    'Address',
    'Comments',
    'OSEBuildingID' # it's the primary key
]

# Redundant columns
redundant_columns = [
    'ListOfAllPropertyUseTypes', # tough to encode
    'LargestPropertyUseType',
    'LargestPropertyUseTypeGFA', # redundant with GFA Building
    'SecondLargestPropertyUseType',
    'SecondLargestPropertyUseTypeGFA',
    'ThirdLargestPropertyUseType',
    'ThirdLargestPropertyUseTypeGFA',
    # Unity Redundancy (We'll keep kBtu, so we'll remove kwh)
    'Electricity(kWh)',
    'NaturalGas(therms)'
]

# The targets
target = [
    'SiteEnergyUse(kBtu)', # The target for the energetic consumption
    'TotalGHGEmissions' # The target for the greenhouse gases emission
]

# The features we'll use for the training of the model
critical_features = [
    'PrimaryPropertyType',
    'YearBuilt',
    'PropertyGFATotal',
    'PropertyGFAParking',
    'Neighborhood',
    'NumberofBuildings',
    'NumberofFloors',
    'BuildingType',
    'Latitude',
    'Longitude'
]
