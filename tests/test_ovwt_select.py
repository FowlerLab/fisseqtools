from typing import Optional

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.linear_model
import sklearn.metrics

from fisseqtools.ovwt_select import ovwt_select, ovwt_select_xgboost_reg


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

    ovwt_select(
        train_fun,
        train_df_path=train_file,
        eval_df_path=eval_file,
        features_path=features_file,
        output_path=output_path,
        select_key="label",
        wt_value="A",
    )

    model_file = output_path / "ovwt_model.pkl"
    metrics_file = output_path / "metrics.csv"

    assert model_file.exists()
    assert metrics_file.exists()

    metrics_df = pd.read_csv(metrics_file)
    print(metrics_df)
    assert "label" in metrics_df.columns
    assert "auc_roc" in metrics_df.columns
    assert "accuracy" in metrics_df.columns
    assert metrics_df.loc[metrics_df["label"] == "A", "auc_roc"].isna().all()
    assert metrics_df.loc[metrics_df["label"] == "A", "accuracy"].isna().all()
    assert all(metrics_df.loc[metrics_df["label"].isin(["B", "C"]), "auc_roc"] == 1.0)
    assert all(metrics_df.loc[metrics_df["label"].isin(["B", "C"]), "accuracy"] == 1.0)


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

    for i, cli_fun in enumerate([ovwt_select_xgboost_reg]):
        output_path = output_path_base / str(i)
        model_file = output_path / "ovwt_model.pkl"
        metrics_file = output_path / "metrics.csv"
        output_path.mkdir()

        assert (
            cli_fun()(
                train_df_path=train_file,
                eval_df_path=eval_file,
                features_path=features_file,
                output_path=output_path,
                select_key="label",
                wt_value="A",
            )
            == 1.0
        )

        assert model_file.exists()
        assert metrics_file.exists()

        metrics_df = pd.read_csv(metrics_file)
        assert "label" in metrics_df.columns
        assert "auc_roc" in metrics_df.columns
        assert "accuracy" in metrics_df.columns
        assert metrics_df.loc[metrics_df["label"] == "A", "auc_roc"].isna().all()
        assert metrics_df.loc[metrics_df["label"] == "A", "accuracy"].isna().all()
        assert all(
            metrics_df.loc[metrics_df["label"].isin(["B", "C"]), "auc_roc"] == 1.0
        )
        assert all(
            metrics_df.loc[metrics_df["label"].isin(["B", "C"]), "accuracy"] == 1.0
        )
