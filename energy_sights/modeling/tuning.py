import numpy as np
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV
from energy_sights.logging_config import setup_logging, get_task_logger
from datetime import datetime

setup_logging()

log = get_task_logger(task_name='model_tuning')

def get_param_grid(model_type: str) -> Dict[str, Any]:

    if model_type.strip().lower() == 'random_forest':

        return {
            'regressor__model__n_estimators': [100, 200, 300, 500],
            'regressor__model__max_depth': [None, 10, 20, 30],
            'regressor__model__min_samples_split': [2, 5, 10],
            'regressor__model__min_samples_leaf': [1, 2, 4]
        }

    elif model_type.strip().lower() == 'xgboost':
        return {
            'regressor__model__n_estimators': [100, 200, 500],
            'regressor__model__max_depth': [3, 5, 7, 9],
            'regressor__model__learning_rate': [0.01, 0.02, 0.07, 0.1],
            'regressor__model__subsample': [0.6, 0.8, 1.0],
            'regressor__model__colsample_bytree': [0.6, 0.8, 1.0]
        }


    return {}


def tune_model(model, X_train, y_train, model_type: str, n_iter: int=20):

    param_grid = get_param_grid(model_type=model_type)

    log.info(f'Start of the tuning the: {datetime.now()}')

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        n_jobs=-1, # this use all the core of the CPU
        verbose=2.2,
        random_state=42
    )

    search.fit(X_train, y_train)

    print(f"The best models finds are: {search.best_params_}")
    print(f"The best Score (RMSE roughly): {np.sqrt(-search.best_score_):.2f}")

    return search.best_estimator_
