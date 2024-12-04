import json
from typing import Optional

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.linear_model
import sklearn.metrics

from fisseqtools.ovr_select import (
    ovr_hyperparam_search,
    ovr_select,
    ovr_select_log,
    ovr_select_xgboost,
    ovr_select_xgboost_reg,
    ovr_xgb_search_height,
    train_log_regression,
    train_xgboost,
    train_xgboost_reg,
)


def test_train_log_regression():
    x_train = np.array([[1, 0, 0]] * 50 + [[0, 1, 0]] * 50).astype(float)
    y_train = np.array([0] * 50 + [1] * 50)
    x_eval = np.array([[1, 0, 0]] * 20 + [[0, 1, 0]] * 20).astype(float)
    y_eval = np.array([0] * 20 + [1] * 20)

    model = train_log_regression(x_train, y_train, x_eval, y_eval)
    y_probs = model.predict_proba(x_eval)[:, 1].flatten()
    y_pred = model.predict(x_eval)

    assert sklearn.metrics.roc_auc_score(y_eval, y_probs) == 1.0
    assert sklearn.metrics.accuracy_score(y_eval, y_pred) == 1.0


def test_train_xgboost():
    x_train = np.array([[1, 0, 0]] * 50 + [[0, 1, 0]] * 50).astype(float)
    y_train = np.array([0] * 50 + [1] * 50)
    x_eval = np.array([[1, 0, 0]] * 20 + [[0, 1, 0]] * 20).astype(float)
    y_eval = np.array([0] * 20 + [1] * 20)

    model = train_xgboost(x_train, y_train, x_eval, y_eval)
    y_probs = model.predict_proba(x_eval)[:, 1].flatten()
    y_pred = model.predict(x_eval)

    assert sklearn.metrics.roc_auc_score(y_eval, y_probs) == 1.0
    assert sklearn.metrics.accuracy_score(y_eval, y_pred) == 1.0


def test_train_xgboost_reg():
    x_train = np.array([[1, 0, 0]] * 50 + [[0, 1, 0]] * 50).astype(float)
    y_train = np.array([0] * 50 + [1] * 50)
    x_eval = np.array([[1, 0, 0]] * 20 + [[0, 1, 0]] * 20).astype(float)
    y_eval = np.array([0] * 20 + [1] * 20)

    model = train_xgboost_reg(x_train, y_train, x_eval, y_eval)
    y_probs = model.predict_proba(x_eval)[:, 1].flatten()
    y_pred = model.predict(x_eval)

    assert sklearn.metrics.roc_auc_score(y_eval, y_probs) == 1.0
    assert sklearn.metrics.accuracy_score(y_eval, y_pred) == 1.0


def test_ovr_select(tmp_path):
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "index": [0] * 50 + [1] * 50 + [2] * 50,
        }
    )
    eval_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [2] * 20,
        }
    )

    features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)
    train_file = tmp_path / "train.csv"
    eval_file = tmp_path / "eval.csv"
    features_file = tmp_path / "features.npy"
    output_path = tmp_path / "output"

    # Save input data
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    np.save(features_file, features)
    output_path.mkdir()

    def train_fun(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        sample_weight: Optional[np.ndarray | None] = None,
    ) -> sklearn.base.BaseEstimator:
        return sklearn.linear_model.LogisticRegression().fit(
            x_train, y_train, sample_weight=sample_weight
        )

    ovr_select(
        train_fun,
        train_df_path=train_file,
        eval_df_path=eval_file,
        features_path=features_file,
        output_path=output_path,
        select_key="label",
    )

    model_file = output_path / "ovr_model.pkl"
    metrics_file = output_path / "metrics.csv"
    predictions_file = output_path / "predictions.csv"

    assert model_file.exists()
    assert metrics_file.exists()
    assert predictions_file.exists()

    metrics_df = pd.read_csv(metrics_file)
    print(metrics_df)
    assert "label" in metrics_df.columns
    assert "auc_roc" in metrics_df.columns
    assert "accuracy" in metrics_df.columns

    predictions_df = pd.read_csv(predictions_file)
    assert "true_label" in predictions_df.columns
    assert "label_predicted" in predictions_df.columns

    assert all(metrics_df["auc_roc"] == 1.0)
    assert all(metrics_df["accuracy"] == 1.0)


def test_over_hyperparam_search(tmp_path):
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "index": [0] * 50 + [1] * 50 + [2] * 50,
        }
    )
    eval_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [2] * 20,
        }
    )
    features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)
    train_file = tmp_path / "train.csv"
    eval_file = tmp_path / "eval.csv"
    features_file = tmp_path / "features.npy"
    output_path = tmp_path / "output"

    # Save input data
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    np.save(features_file, features)
    output_path.mkdir()

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
    }

    def train_fun(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        sample_weight: Optional[np.ndarray | None] = None,
        C: float = 1.0,
        penalty: str = "l1",
    ) -> sklearn.base.BaseEstimator:
        return sklearn.linear_model.LogisticRegression(C=C, penalty=penalty).fit(
            x_train, y_train, sample_weight=sample_weight
        )

    best_params = ovr_hyperparam_search(
        train_fun=train_fun,
        param_grid=param_grid,
        train_df_path=train_file,
        eval_df_path=eval_file,
        features_path=features_file,
        output_path=output_path,
        select_key="label",
        class_sample=2,
    )

    assert isinstance(best_params, dict)
    assert "C" in best_params
    assert best_params["C"] in param_grid["C"]
    assert best_params["penalty"] == "l2"

    best_params_file = output_path / "best_params.json"
    assert best_params_file.exists()

    with open(best_params_file, "r") as f:
        saved_params = json.load(f)

    assert saved_params == best_params


def test_cli_train_wrappers(tmp_path):
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "index": [0] * 50 + [1] * 50 + [2] * 50,
        }
    )
    eval_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [2] * 20,
        }
    )

    features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)
    train_file = tmp_path / "train.csv"
    eval_file = tmp_path / "eval.csv"
    features_file = tmp_path / "features.npy"
    output_path_base = tmp_path / "output"
    output_path_base.mkdir()

    # Save input data
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    np.save(features_file, features)

    for i, cli_fun in enumerate(
        [ovr_select_log, ovr_select_xgboost, ovr_select_xgboost_reg]
    ):
        output_path = output_path_base / str(i)
        model_file = output_path / "ovr_model.pkl"
        metrics_file = output_path / "metrics.csv"
        predictions_file = output_path / "predictions.csv"
        output_path.mkdir()

        assert (
            cli_fun()(
                train_df_path=train_file,
                eval_df_path=eval_file,
                features_path=features_file,
                output_path=output_path,
                select_key="label",
            )
            == 1.0
        )

        assert model_file.exists()
        assert metrics_file.exists()
        assert predictions_file.exists()

        metrics_df = pd.read_csv(metrics_file)
        print(metrics_df)
        assert "label" in metrics_df.columns
        assert "auc_roc" in metrics_df.columns
        assert "accuracy" in metrics_df.columns

        predictions_df = pd.read_csv(predictions_file)
        assert "true_label" in predictions_df.columns
        assert "label_predicted" in predictions_df.columns

        assert all(metrics_df["auc_roc"] == 1.0)
        assert all(metrics_df["accuracy"] == 1.0)


def test_hparam_cli_entries(tmp_path):
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "index": [0] * 50 + [1] * 50 + [2] * 50,
        }
    )
    eval_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [2] * 20,
        }
    )
    features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)
    train_file = tmp_path / "train.csv"
    eval_file = tmp_path / "eval.csv"
    features_file = tmp_path / "features.npy"
    output_base = tmp_path / "output"

    # Save input data
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    np.save(features_file, features)
    output_base.mkdir()

    for i, ovr_hparam_fun in enumerate([ovr_xgb_search_height]):
        output_path = output_base / str(i)
        output_path.mkdir()

        best_params = ovr_hparam_fun()(
            train_df_path=train_file,
            eval_df_path=eval_file,
            features_path=features_file,
            output_path=output_path,
            select_key="label",
            class_sample=2,
        )

        assert isinstance(best_params, dict)
        best_params_file = output_path / "best_params.json"
        assert best_params_file.exists()

        with open(best_params_file, "r") as f:
            saved_params = json.load(f)

        assert saved_params == best_params
