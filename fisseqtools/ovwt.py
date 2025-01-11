from typing import Callable, Optional

import pandas as pd
import xgboost as xgb
import numpy as np
import sklearn

TrainFun = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray | None]],
    sklearn.base.BaseEstimator,
]


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        colsample_bynode=0.7,
        early_stopping_rounds=10,
        n_estimators=100,
        eval_metric="auc",
    ).fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_eval, y_eval)],
        sample_weight=sample_weight,
        verbose=True,
    )

def train_ovwt(
    train_fun: TrainFun,
    train_split: pd.DataFrame,
    eval_one_split: pd.DataFrame,
    eval_two_split: pd.DataFrame,
) -> dict[str, sklearn.base.BaseEstimator]:
    pass