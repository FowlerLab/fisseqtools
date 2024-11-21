import json
import pickle

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from fisseqtools.xgboost_select import (
    compute_metrics,
    xgboost_select,
    search_hyperparams,
)


# Sample data for testing
@pytest.fixture
def sample_data():
    labels = pd.Series(["A", "A", "B", "B", "B", "C", "C", "C", "C", "D"])
    embeddings = np.random.rand(10, 5)  # 10 samples, 5 features
    frequency_cutoff = 3
    return labels, embeddings, frequency_cutoff


@pytest.fixture
def split_sample_data():
    embeddings = np.random.rand(100, 10)
    labels = np.random.choice([0, 1, 2], size=100)  # Three classes
    return embeddings, labels


def test_compute_metrics(metrics_sample_data):
    x_test, y_test, y_pred, label_encoder = metrics_sample_data

    class MockModel:
        def predict(self, X):
            return y_pred

        def predict_proba(self, X):
            return np.array(
                [
                    [0.7 if pred == label else 0.15 for label in label_encoder.classes_]
                    for pred in y_pred
                ]
            )

    model = MockModel()
    auc_roc_series, accuracy_series, label_true, label_pred = compute_metrics(
        model, x_test, y_test, label_encoder
    )

    assert np.allclose(label_true, label_encoder.inverse_transform(y_test))
    assert np.allclose(
        label_pred, label_encoder.inverse_transform(model.predict(x_test))
    )

    accuracies = np.empty((len(label_encoder.classes_),))
    for label in label_encoder.classes_:
        label_mask = y_test == label
        accuracies[label] = np.mean((y_pred == y_test)[label_mask])

    true_accuracy_series = pd.Series(accuracies, index=label_encoder.classes_)

    assert isinstance(auc_roc_series, pd.Series)
    assert isinstance(accuracy_series, pd.Series)

    assert set(auc_roc_series.index) == set(
        label_encoder.inverse_transform(np.unique(y_test))
    )
    assert set(accuracy_series.index) == set(
        label_encoder.inverse_transform(np.unique(y_test))
    )

    assert (auc_roc_series >= 0).all() and (auc_roc_series <= 1).all()
    assert true_accuracy_series.equals(accuracy_series)


def test_xgboost_select(tmp_path):
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
    test_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": [0] * 20 + [1] * 20 + [2] * 20,
        }
    )
    features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)

    train_file = tmp_path / "train.csv"
    eval_file = tmp_path / "eval.csv"
    test_file = tmp_path / "test.csv"
    features_file = tmp_path / "features.npy"
    output_path = tmp_path / "output"

    # Save input data
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    test_df.to_csv(test_file, index=False)
    np.save(features_file, features)
    output_path.mkdir()

    xgboost_select(
        train_df_path=train_file,
        eval_df_path=eval_file,
        test_df_path=test_file,
        features_path=features_file,
        output_path=output_path,
        select_key="label",
    )

    model_file = output_path / "xgboost_model.pkl"
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


def test_search_hyperparams(tmp_path):
    train_df = pd.DataFrame(
        {
            "label": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "index": list(range(150)),
        }
    )
    eval_df = pd.DataFrame(
        {
            "label": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "index": list(range(150, 210)),
        }
    )
    features = np.random.rand(210, 5)

    # Save input data to temporary files
    train_file = tmp_path / "train.csv"
    eval_file = tmp_path / "eval.csv"
    features_file = tmp_path / "features.npy"
    output_path = tmp_path / "output"

    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    np.save(features_file, features)
    output_path.mkdir()

    search_hyperparams(
        train_df_path=train_file,
        eval_df_path=eval_file,
        features_path=features_file,
        output_path=output_path,
        select_key="label",
        n_testing_rounds=5,
    )

    model_file = output_path / "best_xgboost_model.pkl"
    params_file = next(output_path.glob("best_xgboost_params_*.json"), None)
    assert model_file.exists()
    assert params_file is not None

    with open(params_file, "r") as f:
        best_params = json.load(f)
        # assert "learning_rate" in best_params
        assert "max_depth" in best_params

    with open(model_file, "rb") as f:
        model = pickle.load(f)
        assert hasattr(model, "predict")
