import functools
import os
import pathlib
import pickle
from typing import List, Tuple, Callable

import fire
import numpy as np
import pandas as pd
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.multiclass
import sklearn.preprocessing
import sklearn.utils

from .ovr_select import TrainFun, compute_metrics, train_xgboost_reg
from ..utils import save_metrics_no_prediction


def train_ovwt_model(
    train_fun: TrainFun,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    wt_label: int,
) -> Tuple[List[sklearn.base.BaseEstimator], np.ndarray, np.ndarray]:
    unique_labels = np.unique(y_train)
    classifiers = []
    roc_auc = np.empty_like(unique_labels, dtype=float)
    accuracy = np.empty_like(unique_labels, dtype=float)

    for curr_label in np.unique(y_train):
        if curr_label == wt_label:
            print("Wild Type Label, Skipping", flush=True)
            print("", flush=True)
            roc_auc[curr_label] = np.nan
            accuracy[curr_label] = np.nan

            continue

        print(
            f"Training classifier {curr_label + 1} of {unique_labels.max() + 1}",
            flush=True,
        )

        x_train_filtered = x_train[(y_train == curr_label) | (y_train == wt_label)]
        y_train_filtered = y_train[(y_train == curr_label) | (y_train == wt_label)]
        x_eval_filtered = x_eval[(y_eval == curr_label) | (y_eval == wt_label)]
        y_eval_filtered = y_eval[(y_eval == curr_label) | (y_eval == wt_label)]

        curr_y_train = np.zeros_like(y_train_filtered, dtype=int)
        curr_y_train[y_train_filtered == curr_label] = 1
        curr_y_eval = np.zeros_like(y_eval_filtered, dtype=int)
        curr_y_eval[y_eval_filtered == curr_label] = 1
        weights = sklearn.utils.compute_sample_weight("balanced", curr_y_train)
        next_classifier = train_fun(
            x_train_filtered, curr_y_train, x_eval_filtered, curr_y_eval, weights
        )

        y_probs = next_classifier.predict_proba(x_eval_filtered)[:, 1].flatten()
        y_pred = next_classifier.predict(x_eval_filtered)
        roc_auc[curr_label] = sklearn.metrics.roc_auc_score(curr_y_eval, y_probs)
        accuracy[curr_label] = sklearn.metrics.accuracy_score(curr_y_eval, y_pred)
        y_probs_train = next_classifier.predict_proba(x_train_filtered)[:, 1].flatten()
        train_roc_auc = sklearn.metrics.roc_auc_score(curr_y_train, y_probs_train)
        classifiers.append(next_classifier)

        print(f"Label {curr_label} ROC-AUC: {roc_auc[curr_label]}", flush=True)
        print(f"Label {curr_label} ROC-AUC (Train): {train_roc_auc}", flush=True)
        print("", flush=True)

    return classifiers, roc_auc, accuracy


def ovwt_select(
    train_fun: TrainFun,
    train_df_path: os.PathLike,
    eval_df_path: os.PathLike,
    features_path: os.PathLike,
    output_path: os.PathLike,
    select_key: str = "aaChanges",
    wt_value: str = "WT",
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
    wt_label = label_encoder.transform([wt_value])[0]

    classifiers, roc_auc, accuracy = train_ovwt_model(
        train_fun,
        x_train,
        y_train,
        x_eval,
        y_eval,
        wt_label,
    )

    with open(output_path / "ovwt_model.pkl", "wb") as f:
        pickle.dump(classifiers, f)

    auc_roc_series = pd.Series(
        roc_auc,
        index=label_encoder.inverse_transform(np.unique(y_eval)),
    )
    accuracy_series = pd.Series(
        accuracy,
        index=label_encoder.inverse_transform(np.unique(y_eval)),
    )

    # Save Metrics
    save_metrics_no_prediction(
        train_df,
        auc_roc_series,
        accuracy_series,
        select_key,
        output_path,
    )

    return roc_auc[~np.isnan(roc_auc)].mean()


def ovwt_select_xgboost_reg() -> Callable:
    return functools.partial(ovwt_select, train_xgboost_reg)


if __name__ == "__main__":
    fire.Fire({"ovwt_xgb_reg_select": ovwt_select_xgboost_reg()})
