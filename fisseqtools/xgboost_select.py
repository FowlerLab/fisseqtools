import os
import pathlib
import pickle
from typing import Tuple, Iterable

import fire
import numpy as np
import pandas as pd
import sklearn.experimental.enable_halving_search_cv
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils.class_weight
import xgboost as xgb


def filter_labels(
    labels: pd.Series, embeddings: np.ndarray, frequency_cutoff: int
) -> Tuple[pd.Series, np.ndarray]:
    label_counts = labels.value_counts()
    freq_mask = labels.map(label_counts) >= frequency_cutoff
    valid_labels = labels[freq_mask]
    valid_embeddings = embeddings[freq_mask.to_numpy()]
    return valid_labels, valid_embeddings


def split_data(
    embeddings: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        x_train,
        x_eval_test,
        y_train,
        y_eval_test,
    ) = sklearn.model_selection.train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels
    )
    x_eval, x_test, y_eval, y_test = sklearn.model_selection.train_test_split(
        x_eval_test, y_eval_test, test_size=0.5, stratify=y_eval_test
    )
    return x_train, x_eval, x_test, y_train, y_eval, y_test


def train_model(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray
) -> xgb.XGBClassifier:
    sample_weights = sklearn.utils.compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )
    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        early_stopping_rounds=5,
    )
    xgb_clf.fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(x_eval, y_eval)],
        verbose=True,
    )
    return xgb_clf


def compute_metrics(
    model: xgb.XGBClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: sklearn.preprocessing.LabelEncoder,
) -> Tuple[pd.Series, pd.Series, Iterable[str], Iterable[str]]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    y_test_binarized = sklearn.preprocessing.label_binarize(
        y_test, classes=np.unique(y_test)
    )

    label_auc_roc = sklearn.metrics.roc_auc_score(
        y_test_binarized, y_prob, average=None
    )
    n_unique_classes = len(np.unique(y_test))
    auc_roc_series = pd.Series(
        label_auc_roc,
        index=label_encoder.inverse_transform(np.arange(n_unique_classes)),
    )

    is_correct = (y_pred == y_test).astype(int)
    test_labels = label_encoder.inverse_transform(y_test)
    accuracy_df = pd.DataFrame({"label": test_labels, "is_correct": is_correct})
    accuracy_series = accuracy_df.groupby("label")["is_correct"].mean()

    return (
        auc_roc_series,
        accuracy_series,
        test_labels,
        label_encoder.inverse_transform(y_pred),
    )


def save_metrics(
    data_df: pd.DataFrame,
    auc_roc_series: pd.Series,
    accuracy_series: pd.Series,
    select_key: str,
    output_path: pathlib.Path,
    label_true: Iterable[str],
    label_pred: Iterable[str],
) -> None:
    metrics_df = pd.DataFrame({"label": data_df[select_key].unique()})
    metrics_df["auc_roc"] = metrics_df["label"].map(auc_roc_series)
    metrics_df["accuracy"] = metrics_df["label"].map(accuracy_series)
    metrics_df.to_csv(output_path / "metrics.csv", index=False)
    pd.DataFrame({"true_label": label_true, "label_predicted": label_pred}).to_csv(
        output_path / "predictions.csv"
    )


def xgboost_select(
    data_df_path: os.PathLike,
    embeddings_pkl_path: os.PathLike,
    output_path: os.PathLike,
    select_key: str,
    frequency_cutoff: int,
) -> None:
    output_path = pathlib.Path(output_path)
    data_df = pd.read_csv(data_df_path)
    with open(embeddings_pkl_path, "rb") as f:
        embeddings = pickle.load(f)
    labels = data_df[select_key]

    # Load and preprocess data
    valid_labels, valid_embeddings = filter_labels(labels, embeddings, frequency_cutoff)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(valid_labels)
    encoded_labels = label_encoder.transform(valid_labels)

    # Split data
    x_train, x_eval, x_test, y_train, y_eval, y_test = split_data(
        valid_embeddings, encoded_labels
    )

    # Train model
    xgb_clf = train_model(x_train, y_train, x_eval, y_eval)
    with open(output_path / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_clf, f)

    # Compute and save metrics
    auc_roc_series, accuracy_series, label_true, label_pred = compute_metrics(
        xgb_clf, x_test, y_test, label_encoder
    )
    save_metrics(
        data_df,
        auc_roc_series,
        accuracy_series,
        select_key,
        output_path,
        label_true,
        label_pred,
    )


if __name__ == "__main__":
    fire.Fire({"xgboost_select": xgboost_select})
