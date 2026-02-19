import numpy as np
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV, KFold
from energy_sights.logging_config import setup_logging, get_task_logger
from datetime import datetime
from numpy.typing import NDArray
from pandas import DataFrame

setup_logging()

log = get_task_logger(task_name='model_tuning')

def get_param_grid(model_type: str) -> Dict[str, Any]:

    if model_type.strip().lower() == 'random_forest':

        return {
            'regressor__model__n_estimators': [200, 300, 500],
            'regressor__model__max_depth': [5, 7, 10],
            'regressor__model__min_samples_split': [10, 20, 30],
            'regressor__model__min_samples_leaf': [8, 10, 12, 5],
            'regressor__model__bootstrap': [True],
            'regressor__model__max_features': ['sqrt', .5, .7]

        }

    elif model_type.strip().lower() == 'xgboost':
        return {
            'regressor__model__n_estimators': [300, 500, 200],
            'regressor__model__max_depth': [4, 6, 8],
            'regressor__model__learning_rate': [0.01, 0.03, 0.05],
            'regressor__model__subsample': [0.8, 0.7],
            'regressor__model__colsample_bytree': [0.6, 0.5, 0.7],
            'regressor__model__gamma': [0.5, 0.1, 0],
            'regressor__model__min_child_weight': [1, 3, 5],
            # regularization to avoid the overfitting of the model
            'regressor__model__alpha': [0.8, 5, 1.0], # L1 Lasso
            'regressor__model__lambda': [1, 2.0, 3.0]
        }


    return {}


def tune_model(model, X_train: NDArray | DataFrame, y_train: NDArray | DataFrame, model_type: str, n_iter: int=20):
    cv_strategy = KFold(n_splits=5, random_state=42, shuffle=True)
    param_grid = get_param_grid(model_type=model_type)

    log.info(f'Start of the tuning the: {datetime.now()}')

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=cv_strategy,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, # this use all the core of the CPU
        verbose=2,
        random_state=42,
        return_train_score=True
    )
    log.info(f'Start of the training for {model_type}')

    search.fit(X_train, y_train)
    # let's avoid the overfitting right now
    best_idx = search.best_index_
    train_score = -search.cv_results_['mean_train_score'][best_idx]
    val_score = -search.cv_results_['mean_test_score'][best_idx]

    print(f"The best parameters finds are: {search.best_params_}")
    print(f"The MAE on the train set: {train_score: .2f}")
    print(f"The MAE on the validation set: {val_score: .2f}")
    print(f"The best Score (RMSE roughly): {np.sqrt(-search.best_score_):.2f}")
    print(f"Gap (Overfitting): {abs(train_score - val_score): .2f}")

    return search.best_estimator_
