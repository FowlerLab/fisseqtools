import json
import os
import pathlib
import pickle
from typing import Tuple, Iterable, Optional

import fire
import numpy as np
import pandas as pd
import sklearn.experimental.enable_halving_search_cv
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils.class_weight
import xgboost as xgb


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
    train_df_path: os.PathLike,
    eval_df_path: os.PathLike,
    test_df_path: os.PathLike,
    features_path: os.PathLike,
    output_path: os.PathLike,
    select_key: str,
    learning_rate: Optional[float] = 0.1,
    n_estimators: Optional[int] = 100,
    max_depth: Optional[int] = 3,
) -> None:
    output_path = pathlib.Path(output_path)
    train_df = pd.read_csv(train_df_path)
    eval_df = pd.read_csv(eval_df_path)
    test_df = pd.read_csv(test_df_path)
    features = np.load(features_path)

    x_train = features[train_df["index"]]
    x_eval = features[eval_df["index"]]
    x_test = features[test_df["index"]]

    labels = train_df[select_key]
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(labels)
    y_train = label_encoder.transform(train_df[select_key])
    y_eval = label_encoder.transform(eval_df[select_key])
    y_test = label_encoder.transform(test_df[select_key])

    # Train model
    xgb_clf = xgb.XGBClassifier(
        eval_metric="mlogloss",
        early_stopping_rounds=5,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
    ).fit(
        x_train,
        y_train,
        eval_set=[(x_eval, y_eval)],
        verbose=True,
    )

    with open(output_path / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_clf, f)

    # Compute and save metrics
    auc_roc_series, accuracy_series, label_true, label_pred = compute_metrics(
        xgb_clf, x_test, y_test, label_encoder
    )
    save_metrics(
        train_df,
        auc_roc_series,
        accuracy_series,
        select_key,
        output_path,
        label_true,
        label_pred,
    )


def search_hyperparams(
    train_df_path: os.PathLike,
    eval_df_path: os.PathLike,
    features_path: os.PathLike,
    output_path: os.PathLike,
    select_key: str,
    n_testing_rounds=20,
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

    param_dist = {
        "learning_rate": [0.1, 0.01],
        "max_depth": [3, 6, 9, 12],
    }

    best_params = None
    best_score = -float("inf")
    best_model = None
    for params in sklearn.model_selection.ParameterGrid(param_dist):
        print(f"Testing parameters: {params}")
        model = xgb.XGBClassifier(
            eval_metric="mlogloss",
            learning_rate=params["learning_rate"],
            n_estimators=n_testing_rounds,
            max_depth=params["max_depth"],
        ).fit(
            x_train,
            y_train,
            verbose=True,
        )

        curr_score = model.score(x_eval, y_eval)
        if curr_score > best_score:
            best_score = curr_score
            best_params = params
            best_model = model

    with open(output_path / "best_xgboost_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(output_path / f"best_xgboost_params_{best_score:.2f}.json", "w") as f:
        json.dump(best_params, f)


if __name__ == "__main__":
    fire.Fire({"xgboost_select": xgboost_select, "hyperparams": search_hyperparams})
