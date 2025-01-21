import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble
import sklearn.linear_model

from fisseqtools.ovwt import (
    get_mask_features,
    get_metrics,
    get_shap_values,
    get_train_data_labels,
    ovwt,
    ovwt_shap_only,
    train_ovwt,
    train_xgboost,
)


def test_train_xgboost():
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50,
            "index": [0] * 50 + [1] * 50,
        }
    )
    eval_one_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20,
            "index": [0] * 20 + [1] * 20,
        }
    )

    model = train_xgboost(
        train_df["index"].to_numpy().reshape((-1, 1)),
        train_df["index"].to_numpy(),
        eval_one_df["index"].to_numpy().reshape((-1, 1)),
        eval_one_df["index"].to_numpy(),
    )

    train_predictions = model.predict(train_df["index"].to_numpy().reshape((-1, 1)))
    eval_predictons = model.predict(eval_one_df["index"].to_numpy().reshape((-1, 1)))

    assert np.array_equal(train_predictions, train_df["index"].to_numpy())
    assert np.array_equal(eval_predictons, eval_one_df["index"].to_numpy())


def test_get_mask_features():
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50,
            "index": [0] * 50 + [1] * 50,
        }
    )

    wt_mask, feature_matrix = get_mask_features("label", ["index"], "B", train_df)

    assert np.array_equal(wt_mask, train_df["index"].to_numpy(dtype=bool))
    assert np.array_equal(feature_matrix, train_df["index"].to_numpy().reshape((-1, 1)))
    assert wt_mask.dtype == np.bool
    assert feature_matrix.dtype == np.float64


def test_get_train_data_labels():
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50,
            "index": [0] * 50 + [1] * 50,
        }
    )

    wt_mask = train_df["index"].to_numpy(dtype=bool)
    variant_mask = ~wt_mask
    feature_matrix = train_df["index"].to_numpy().reshape((-1, 1))
    features, labels = get_train_data_labels(wt_mask, variant_mask, feature_matrix)
    expected_features = np.array([[1]] * 50 + [[0]] * 50)

    assert np.array_equal(labels, wt_mask)
    assert np.array_equal(features, expected_features)


def test_get_metrics():
    train_df = pd.DataFrame(
        {
            "label": [1] * 50 + [0] * 50,
            "index": [0] * 50 + [1] * 50,
        }
    )

    model = sklearn.linear_model.LogisticRegression()
    model.fit(
        train_df["index"].to_numpy().reshape((-1, 1)),
        train_df["label"].to_numpy(dtype=bool),
    )

    roc_auc, accuracy = get_metrics(
        model,
        train_df["index"].to_numpy().reshape((-1, 1)),
        train_df["label"].to_numpy(dtype=bool),
    )

    assert roc_auc == pytest.approx(1, 0)
    assert accuracy == pytest.approx(1.0)


def test_train_ovwt():
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "index": [0] * 50 + [1] * 50 + [1] * 50,
        }
    )
    eval_one_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [1] * 20,
        }
    )
    eval_two_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [1] * 20,
        }
    )

    def train_fun(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        sample_weight: Optional[np.ndarray | None] = None,
    ) -> sklearn.base.BaseEstimator:
        return sklearn.linear_model.LogisticRegression().fit(x_train, y_train)

    models, metrics = train_ovwt(
        train_fun,
        train_df,
        eval_one_df,
        eval_two_df,
        {"target_column": "label", "feature_columns": ["index"]},
        wt_key="A",
    )

    assert "A" not in models
    assert "B" in models
    assert "C" in models

    assert metrics.shape == (2, 7)
    assert "B" in set(metrics["label"])
    assert "C" in set(metrics["label"])

    metrics = metrics.drop("label", axis=1)
    assert np.all(metrics.to_numpy() == pytest.approx(1.0))


def test_get_shap_values():
    train_df = pd.DataFrame(
        {
            "label": [0] * 50 + [1] * 50,
            "index": [0] * 50 + [1] * 50,
        }
    )

    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(
        train_df["index"].to_numpy().reshape((-1, 1)),
        train_df["label"].to_numpy(dtype=bool),
    )

    train_df["label"] = ["A"] * 50 + ["B"] * 50
    models = {"B": model}

    shap_vals = get_shap_values(
        train_df,
        models,
        {"target_column": "label", "feature_columns": ["index"]},
        wt_key="A",
    )

    assert shap_vals.shape == (50, 3)
    assert "label" in shap_vals.columns
    assert "index" in shap_vals.columns
    assert all(shap_vals["label"] == "B")
    assert all(shap_vals["p_is_var"] == pytest.approx(1.0))
    assert all(~shap_vals["index"].isna())


def test_ovwt(tmp_path):
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50,
            "index": [0] * 50 + [1] * 50,
        }
    )
    eval_one_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20,
            "index": [0] * 20 + [1] * 20,
        }
    )
    eval_two_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20,
            "index": [0] * 20 + [1] * 20,
        }
    )
    meta_data = {"target_column": "label", "feature_columns": ["index"]}
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    train_df.to_parquet(tmp_path / "train.parquet")
    eval_one_df.to_parquet(tmp_path / "eval_one.parquet")
    eval_two_df.to_parquet(tmp_path / "eval_two.parquet")
    with open(tmp_path / "meta_data.json", "w") as f:
        json.dump(meta_data, f)

    def train_fun(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        sample_weight: Optional[np.ndarray | None] = None,
    ) -> sklearn.base.BaseEstimator:
        return sklearn.ensemble.RandomForestClassifier().fit(x_train, y_train)

    ovwt(
        train_fun,
        str(tmp_path / "train.parquet"),
        str(tmp_path / "eval_one.parquet"),
        str(tmp_path / "eval_two.parquet"),
        str(tmp_path / "meta_data.json"),
        str(output_dir),
        wt_key="A",
    )

    assert pathlib.Path(output_dir / "train_results.csv").is_file()
    assert pathlib.Path(output_dir / "models.pkl").is_file()
    assert pathlib.Path(output_dir / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_one_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_two_shap.parquet").is_file()

    output_dir_shap_only = tmp_path / "shap_only"
    output_dir_shap_only.mkdir()

    ovwt_shap_only(
        str(output_dir / "models.pkl"),
        str(tmp_path / "train.parquet"),
        str(tmp_path / "eval_one.parquet"),
        str(tmp_path / "eval_two.parquet"),
        str(tmp_path / "meta_data.json"),
        str(output_dir_shap_only),
        wt_key="A",
    )

    assert pathlib.Path(output_dir_shap_only / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir_shap_only / "eval_one_shap.parquet").is_file()
    assert pathlib.Path(output_dir_shap_only / "eval_two_shap.parquet").is_file()
