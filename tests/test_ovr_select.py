import pickle

import sklearn.linear_model
import numpy as np
import pandas as pd

from fisseqtools.ovr_select import ovr_select, compute_metrics


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

    ovr_select(
        base_model=sklearn.linear_model.LogisticRegression,
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
    assert "label" in metrics_df.columns
    assert "auc_roc" in metrics_df.columns
    assert "accuracy" in metrics_df.columns

    predictions_df = pd.read_csv(predictions_file)
    assert "true_label" in predictions_df.columns
    assert "label_predicted" in predictions_df.columns

    assert all(metrics_df["auc_roc"] == 1.0)
    assert all(metrics_df["accuracy"] == 1.0)
