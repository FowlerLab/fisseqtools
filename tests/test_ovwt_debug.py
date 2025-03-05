import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble
import sklearn.linear_model

from fisseqtools.ovwt_debug import (
    get_mask_features,
    get_metrics,
    get_shap_values,
    get_train_data_labels,
    ovwt,
    ovwt_single_feature,
    ovwt_shap_only,
    ovwt_stratified,
    train_ovwt,
    train_xgboost,
    train_single_feature_xgboost,
)


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
        train_fun=train_fun,
        train_data_path=str(tmp_path / "train.parquet"),
        eval_one_data_path=str(tmp_path / "eval_one.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir),
        wt_key="A",
        eval_two_data_path=str(tmp_path / "eval_two.parquet"),
    )

    assert pathlib.Path(output_dir / "train_results.csv").is_file()
    assert pathlib.Path(output_dir / "models.pkl").is_file()
    assert pathlib.Path(output_dir / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_two_shap.parquet").is_file()

    output_dir = tmp_path / "no_second_eval"
    output_dir.mkdir()

    ovwt(
        train_fun=train_fun,
        train_data_path=str(tmp_path / "train.parquet"),
        eval_one_data_path=str(tmp_path / "eval_one.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir),
        wt_key="A",
    )

    assert pathlib.Path(output_dir / "train_results.csv").is_file()
    assert pathlib.Path(output_dir / "models.pkl").is_file()
    assert pathlib.Path(output_dir / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_shap.parquet").is_file()
    assert not pathlib.Path(output_dir / "eval_two_shap.parquet").is_file()

    output_dir = tmp_path / "shuffle_labels"
    output_dir.mkdir()

    ovwt(
        train_fun=train_fun,
        train_data_path=str(tmp_path / "train.parquet"),
        eval_one_data_path=str(tmp_path / "eval_one.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir),
        wt_key="A",
        permutate_labels=True,
    )

    assert pathlib.Path(output_dir / "train_results.csv").is_file()
    assert pathlib.Path(output_dir / "models.pkl").is_file()
    assert pathlib.Path(output_dir / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_shap.parquet").is_file()
    assert not pathlib.Path(output_dir / "eval_two_shap.parquet").is_file()

    output_dir = tmp_path / "single_feature"
    output_dir.mkdir()

    ovwt_single_feature(
        train_data_path=str(tmp_path / "train.parquet"),
        eval_one_data_path=str(tmp_path / "eval_one.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir),
        wt_key="A",
    )

    assert pathlib.Path(output_dir / "train_results.csv").is_file()
    assert pathlib.Path(output_dir / "models.pkl").is_file()
    assert pathlib.Path(output_dir / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "eval_shap.parquet").is_file()
    assert pathlib.Path(output_dir / "selected_features.csv").is_file()
    assert not pathlib.Path(output_dir / "eval_two_shap.parquet").is_file()

    output_dir_shap_only = tmp_path / "shap_only"
    output_dir_shap_only.mkdir()

    ovwt_shap_only(
        model_pkl_path=str(output_dir / "models.pkl"),
        train_data_path=str(tmp_path / "train.parquet"),
        eval_one_data_path=str(tmp_path / "eval_one.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir_shap_only),
        eval_two_data_path=str(tmp_path / "eval_two.parquet"),
        wt_key="A",
    )

    assert pathlib.Path(output_dir_shap_only / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir_shap_only / "eval_shap.parquet").is_file()
    assert pathlib.Path(output_dir_shap_only / "eval_two_shap.parquet").is_file()

    output_dir_shap_only = tmp_path / "shap_only_no_second_eval"
    output_dir_shap_only.mkdir()

    ovwt_shap_only(
        model_pkl_path=str(output_dir / "models.pkl"),
        train_data_path=str(tmp_path / "train.parquet"),
        eval_one_data_path=str(tmp_path / "eval_one.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir_shap_only),
        wt_key="A",
    )

    assert pathlib.Path(output_dir_shap_only / "train_shap.parquet").is_file()
    assert pathlib.Path(output_dir_shap_only / "eval_shap.parquet").is_file()
    assert not pathlib.Path(output_dir_shap_only / "eval_two_shap.parquet").is_file()

    input_dir = tmp_path / "stratified"
    input_dir.mkdir()

    train_df_one = train_df.copy()
    train_df_one["replicate"] = 0
    train_df_two = train_df_one.copy()
    train_df_two["replicate"] = 1
    train_df_strat = pd.concat([train_df_one, train_df_two], ignore_index=True)
    train_df_strat.to_parquet(input_dir / "train.parquet")

    eval_df_one = eval_one_df.copy()
    eval_df_one["replicate"] = 0
    eval_df_two = eval_one_df.copy()
    eval_df_two["replicate"] = 1
    eval_df_strat = pd.concat([eval_df_one, eval_df_two], ignore_index=True)
    eval_df_strat.to_parquet(input_dir / "eval.parquet")

    output_dir = tmp_path / "stratified-output"
    output_dir.mkdir()

    ovwt_stratified(
        train_fun=train_fun,
        train_data_path=str(input_dir / "train.parquet"),
        eval_one_data_path=str(input_dir / "eval.parquet"),
        meta_data_json_path=str(tmp_path / "meta_data.json"),
        output_dir=str(output_dir),
        stratify_column="replicate",
        wt_key="A",
    )

    assert pathlib.Path(output_dir / "train_results.csv").is_file()
    assert pathlib.Path(output_dir / "models.pkl").is_file()
