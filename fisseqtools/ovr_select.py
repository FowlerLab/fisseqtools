import functools
import os
import pathlib
import pickle
from typing import List, Iterable, Tuple, Callable

import fire
import sklearn
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.multiclass
import sklearn.preprocessing
import sklearn.utils
import numpy as np
import pandas as pd

from .utils import save_metrics


sklearn.set_config(enable_metadata_routing=True)


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


def ovr_select(
    base_model: type[sklearn.base.BaseEstimator],
    train_df_path: os.PathLike,
    eval_df_path: os.PathLike,
    features_path: os.PathLike,
    output_path: os.PathLike,
    select_key: str,
) -> None:
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

    unique_labels = np.unique(y_train)
    classifiers = []
    roc_auc = np.empty_like(unique_labels)
    accuracy = np.empty_like(unique_labels)

    for curr_label in np.unique(y_train):
        print(
            f"Training classifier {curr_label + 1} of {unique_labels.max() + 1}",
            flush=True,
        )

        curr_y_train = np.zeros_like(y_train, dtype=int)
        curr_y_train[y_train == curr_label] = 1
        weights = sklearn.utils.compute_sample_weight("balanced", curr_y_train)
        next_classifier = base_model().fit(
            x_train, curr_y_train, sample_weight=weights
        )

        curr_y_eval = np.zeros_like(y_eval, dtype=int)
        curr_y_eval[y_eval == curr_label] = 1
        y_probs = next_classifier.predict_proba(x_eval)[:, 1].flatten()
        y_pred = next_classifier.predict(x_eval)
        roc_auc[curr_label] = sklearn.metrics.roc_auc_score(curr_y_eval, y_probs)
        accuracy[curr_label] = sklearn.metrics.accuracy_score(curr_y_eval, y_pred)
        classifiers.append(next_classifier)

        print(f"    Label {curr_label} accuracy: {accuracy[curr_label]}", flush=True)
        print(f"    Label {curr_label} ROC-AUC: {roc_auc[curr_label]}", flush=True)

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


def ovr_select_log() -> Callable:
    return functools.partial(sklearn.linear_model.LogisticRegression)


if __name__ == "__main__":
    fire.Fire({"ovr_log_select": ovr_select_log()})
