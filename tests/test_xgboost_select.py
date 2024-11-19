import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fisseqtools.xgboost_select import compute_metrics, save_metrics, xgboost_select


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


@pytest.fixture
def metrics_sample_data():
    np.random.seed(0)
    x_test = np.random.rand(20, 5)
    y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    label_encoder = LabelEncoder().fit([0, 1, 2])
    return x_test, y_test, y_pred, label_encoder


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


# Test save_metrics function
def test_save_metrics(tmp_path, metrics_sample_data):
    x_test, y_test, y_pred, label_encoder = metrics_sample_data

    auc_roc_series = pd.Series([0.85, 0.75, 0.65], index=label_encoder.classes_)
    accuracy_series = pd.Series([0.90, 0.80, 0.70], index=label_encoder.classes_)
    data_df = pd.DataFrame({"aaChanges": label_encoder.classes_})
    save_metrics(
        data_df,
        auc_roc_series,
        accuracy_series,
        "aaChanges",
        tmp_path,
        ["A", "B"],
        ["B", "A"],
    )

    metrics_file = tmp_path / "metrics.csv"
    assert metrics_file.exists()

    predictions_file = tmp_path / "predictions.csv"
    assert predictions_file.exists()

    test_predictions = pd.read_csv(predictions_file)
    assert test_predictions["true_label"].to_list() == ["A", "B"]
    assert test_predictions["label_predicted"].to_list() == ["B", "A"]

    saved_metrics_df = pd.read_csv(metrics_file)
    assert "label" in saved_metrics_df.columns
    assert "auc_roc" in saved_metrics_df.columns
    assert "accuracy" in saved_metrics_df.columns
    assert saved_metrics_df["auc_roc"].tolist() == auc_roc_series.tolist()
    assert saved_metrics_df["accuracy"].tolist() == accuracy_series.tolist()


def test_train():
    import xgboost as xgb

    xgb.XGBClassifier(
        eval_metric="mlogloss",
        early_stopping_rounds=5,
        num_class=2,
    ).fit(
        np.array([[0, 0], [0, 0], [1, 1], [1, 1]]).astype(float),
        np.array([0, 0, 1, 1]),
        eval_set=[
            (
                np.array([[0, 0], [0, 0], [1, 1], [1, 1]]).astype(float),
                np.array([0, 0, 1, 1]),
            )
        ],
        verbose=True,
    )


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
