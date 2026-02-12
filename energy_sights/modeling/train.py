from jedi.inference.gradual.typing import Callable
from numpy.typing import NDArray

from energy_sights.config import MODELS_DIR
import joblib
import numpy as np
from typing import Dict, Optional
import time
from datetime import datetime
from pathlib import Path

# sklearn models

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator

# logger configuration
from energy_sights.logging_config import setup_logging, get_task_logger
setup_logging()

log = get_task_logger(task_name='model_training')

MODEL_DISPATCHER: Dict[str, str] = {
    'random_forest': RandomForestRegressor,
    'xgboost': XGBRegressor
}

DEFAULT_PARAMS: Dict[str, dict[str, int]] = {
    'random_forest': {
        'n_estimators': 100,
        'n_jobs': -1,
        'max_depth': 15,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_jobs': -1,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }
}

class ModelTrainer:

    def __init__(self, model_type: str, custom_params: Optional[Dict[str, int]] = None):
        self.model_type = model_type

        if self.model_type not in MODEL_DISPATCHER:
            log.error(f"Your model: {model_type} is not in our list of models !")
            raise ValueError(f"Model: {self.model_type} not supported, your choice is: {MODEL_DISPATCHER}.")

        # merge of the default parameter with the custom_params

        self.params = DEFAULT_PARAMS[model_type].copy()
        if custom_params:
            self.params.update(custom_params)

        self.model = None


    def build_pipeline(self, use_log_target: bool = True) -> BaseEstimator:
        regressor_class: BaseEstimator = MODEL_DISPATCHER[self.model_type]
        base_model = regressor_class(**self.params)

        # let's define the encoding for the categorical columns
        te = TargetEncoder(cols=['PrimaryPropertyType', 'Neighborhood'], smoothing=10)

        # let's define hte encoding for the column Neighborhood and BuildingType

        one_hot = OneHotEncoder(cols='BuildingType', handle_unknown='value')

        steps: list[tuple[str, Callable]] = [('target_encoding', te),
                 ('one_hot_encoding', one_hot),
                 ('model', base_model)]

        pipeline = Pipeline(steps)

        # let's define a wrapping for the target

        if use_log_target:
            log.info(f"Model configuration with a logarithm transformation of the target .")

            model_wrapper = TransformedTargetRegressor(
                regressor=pipeline,
                func=np.log, # transform the input
                inverse_func=np.exp # Inverse transformation for the output
            )
            return model_wrapper

        return pipeline

    def train(self, X_train: NDArray, y_train: NDArray, use_log_target: bool=True):
        # start of the training of our model
        log.info(f"Start of the training of the model: {self.model_type} the {datetime.now()}")
        start = time.perf_counter()
        # If the use_log_target is set as True, we don't need to log the target, it will do it for us
        self.model = self.build_pipeline(use_log_target=use_log_target)

        self.model.fit(X_train, y_train)
        end = time.perf_counter()
        log.info(f"End of the training. It takes: {end - start} seconds.")

        return self.model


    def save_model(self, model_name: str) -> None:
        if not self.model:
            log.error("The model is not trained yet !")
            raise ValueError("The model is not trained yet !")

        if not model_name.endswith('pkl'):
            log.error(f"Your model name: {model_name} must have the extension pkl")
            raise ValueError(f"Bad extension for your model {model_name}.")

        save_path: Path = MODELS_DIR / model_name

        joblib.dump(value=self.model, filename=save_path)
        log.info(f"Model saved this: {datetime.now()} at {save_path}")


if __name__ == '__main__':
    pass