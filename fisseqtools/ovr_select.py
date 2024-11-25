import functools
import os
import pathlib
import pickle
from typing import List, Iterable, Tuple, Callable, Optional

import fire
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.preprocessing
import sklearn.utils
import numpy as np
import pandas as pd
import xgboost as xgb

from .utils import save_metrics


sklearn.set_config(enable_metadata_routing=True)

TrainFun = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray | None]],
    sklearn.base.BaseEstimator,
]


def train_log_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    return sklearn.linear_model.LogisticRegression().fit(
        x_train, y_train, sample_weight=sample_weight
    )


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=1,
        colsample_bytree=0.5,
        colsample_bylevel=0.5,
        colsample_bynode=0.5,
        reg_lambda=5,
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


def train_xgboost_reg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    lambda_values = [1, 5, 10]
    best_score = 0.00
    best_model = None
    best_lambda = -1

    for lambda_value in lambda_values:
        print(f"Testing lambda value: {lambda_value}", flush=True)
        next_model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=1,
            colsample_bytree=0.5,
            colsample_bylevel=0.5,
            colsample_bynode=0.5,
            reg_lambda=lambda_value,
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

        curr_score = next_model.best_score
        if curr_score > best_score:
            best_score = curr_score
            best_model = next_model
            best_lambda = lambda_value

    print(f"Best lambda value: {best_lambda}", flush=True)
    return best_model


def compute_metrics(
    classifiers: List[sklearn.base.BaseEstimator],
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: sklearn.preprocessing.LabelEncoder,
) -> Tuple[Iterable[str], Iterable[str]]:
    predict_probas = np.empty((len(y_test), len(np.unique(y_test))))
    for i, classifier in enumerate(classifiers):
        predict_probas[:, i] = classifier.predict_proba(x_test)[:, 1].flatten()

    y_pred = np.argmax(predict_probas, axis=1).flatten()
    return (
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred),
    )


def train_ovr_model(
    train_fun: TrainFun,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> Tuple[List[sklearn.base.BaseEstimator], np.ndarray, np.ndarray]:
    unique_labels = np.unique(y_train)
    classifiers = []
    roc_auc = np.empty_like(unique_labels, dtype=float)
    accuracy = np.empty_like(unique_labels, dtype=float)

    for curr_label in np.unique(y_train):
        print(
            f"Training classifier {curr_label + 1} of {unique_labels.max() + 1}",
            flush=True,
        )

        curr_y_train = np.zeros_like(y_train, dtype=int)
        curr_y_train[y_train == curr_label] = 1
        curr_y_eval = np.zeros_like(y_eval, dtype=int)
        curr_y_eval[y_eval == curr_label] = 1
        weights = sklearn.utils.compute_sample_weight("balanced", curr_y_train)
        next_classifier = train_fun(x_train, curr_y_train, x_eval, curr_y_eval, weights)

        y_probs = next_classifier.predict_proba(x_eval)[:, 1].flatten()
        y_pred = next_classifier.predict(x_eval)
        roc_auc[curr_label] = sklearn.metrics.roc_auc_score(curr_y_eval, y_probs)
        accuracy[curr_label] = sklearn.metrics.accuracy_score(curr_y_eval, y_pred)
        y_probs_train = next_classifier.predict_proba(x_train)[:, 1].flatten()
        train_roc_auc = sklearn.metrics.roc_auc_score(curr_y_train, y_probs_train)
        classifiers.append(next_classifier)

        print(f"Label {curr_label} ROC-AUC: {roc_auc[curr_label]}", flush=True)
        print(f"Label {curr_label} ROC-AUC (Train): {train_roc_auc}", flush=True)
        print("", flush=True)

    return classifiers, roc_auc, accuracy


def ovr_select(
    train_fun: TrainFun,
    train_df_path: os.PathLike,
    eval_df_path: os.PathLike,
    features_path: os.PathLike,
    output_path: os.PathLike,
    select_key: str,
) -> float:
    output_path = pathlib.Path(output_path)
    train_df = pd.read_csv(train_df_path)
    eval_df = pd.read_csv(eval_df_path)
    features = np.load(features_path)

    x_train = features[train_df["index"]]
    x_eval = features[eval_df["index"]]

    labels = train_df[select_key]
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    y_train = label_encoder.transform(train_df[select_key])
    y_eval = label_encoder.transform(eval_df[select_key])

    classifiers, roc_auc, accuracy = train_ovr_model(
        train_fun,
        x_train,
        y_train,
        x_eval,
        y_eval,
    )

    with open(output_path / "ovr_model.pkl", "wb") as f:
        pickle.dump(classifiers, f)

    auc_roc_series = pd.Series(
        roc_auc,
        index=label_encoder.inverse_transform(np.unique(y_eval)),
    )

    accuracy_series = pd.Series(
        accuracy,
        index=label_encoder.inverse_transform(np.unique(y_eval)),
    )

    # Compute and save metrics
    label_true, label_pred = compute_metrics(classifiers, x_eval, y_eval, label_encoder)

    save_metrics(
        train_df,
        auc_roc_series,
        accuracy_series,
        select_key,
        output_path,
        label_true,
        label_pred,
    )

    return roc_auc.mean()


def ovr_select_log() -> Callable:
    return functools.partial(ovr_select, train_log_regression)


def ovr_select_xgboost() -> Callable:
    return functools.partial(ovr_select, train_xgboost)


def ovr_select_xgboost_reg() -> Callable:
    return functools.partial(ovr_select, train_xgboost_reg)


if __name__ == "__main__":
    fire.Fire(
        {
            "ovr_log_select": ovr_select_log(),
            "ovr_xgb_select": ovr_select_xgboost(),
            "ovr_xgb_reg_select": ovr_select_xgboost_reg(),
        }
    )
