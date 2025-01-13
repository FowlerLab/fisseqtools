import functools
import pathlib
from os import PathLike
from typing import Callable, Optional, Dict, List, Optional, Tuple

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


def get_mask_features(
    target_column: str,
    feature_columns: List[str],
    wt_key: str,
    curr_split: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    wt_mask = (curr_split[target_column] == wt_key).to_numpy()
    feature_matrix = curr_split[feature_columns].to_numpy()
    return wt_mask, feature_columns


def train_ovwt(
    train_fun: TrainFun,
    train_split: pd.DataFrame,
    eval_one_split: pd.DataFrame,
    eval_two_split: pd.DataFrame,
    meta_data: Dict[str, str | List[str]],
    wt_key: Optional[str] = "WT",
) -> dict[str, sklearn.base.BaseEstimator]:
    target_column = meta_data["target_column"]
    feature_columns = meta_data["feature_columns"]

    get_features = functools.partial()
    train_wt_mask, train_features

    train_wt_mask = (train_split[target_column] == wt_key).to_numpy()
    eval_one_wt_mask = (eval_one_split[target_column] == wt_key).to_numpy()
    eval_two_wt_mask = (eval_two_split[target_column] == wt_key).to_numpy()

    train_features = train_split[feature_columns].to_numpy()
    eval_one_features = eval_one_split
