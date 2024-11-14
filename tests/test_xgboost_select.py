import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fisseqtools.xgboost_select import (
    filter_labels,
    split_data,
    compute_metrics,
    save_metrics,
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


@pytest.fixture
def metrics_sample_data():
    np.random.seed(0)
    x_test = np.random.rand(20, 5)
    y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    label_encoder = LabelEncoder().fit([0, 1, 2])
    return x_test, y_test, y_pred, label_encoder


# Test filter_labels function
def test_filter_labels(sample_data):
    labels, embeddings, frequency_cutoff = sample_data
    valid_labels, valid_embeddings = filter_labels(labels, embeddings, frequency_cutoff)

    # Check types
    assert isinstance(valid_labels, pd.Series)
    assert isinstance(valid_embeddings, np.ndarray)

    # Check that labels with frequency less than cutoff are removed
    assert all(valid_labels.value_counts() >= frequency_cutoff)


# Test split_data function
def test_split_data(split_sample_data):
    embeddings, labels = split_sample_data
    x_train, x_eval, x_test, y_train, y_eval, y_test = split_data(embeddings, labels)

    # Check shapes
    assert x_train.shape[0] > 0
    assert x_eval.shape[0] > 0
    assert x_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_eval.shape[0] > 0
    assert y_test.shape[0] > 0

    # Check stratification
    assert np.unique(y_train).tolist() == np.unique(labels).tolist()


# Test compute_metrics function
def test_compute_metrics(metrics_sample_data):
    x_test, y_test, y_pred, label_encoder = metrics_sample_data

    # Mock model with preset predictions
    class MockModel:
        def predict(self, X):
            return y_pred

        def predict_proba(self, X):
            # Return random probabilities that match the classes in y_test
            return np.array(
                [
                    [0.7 if pred == label else 0.15 for label in label_encoder.classes_]
                    for pred in y_pred
                ]
            )

    model = MockModel()
    auc_roc_series, accuracy_series = compute_metrics(
        model, x_test, y_test, label_encoder
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
    save_metrics(data_df, auc_roc_series, accuracy_series, "aaChanges", tmp_path)

    metrics_file = tmp_path / "metrics.csv"
    assert metrics_file.exists()

    saved_metrics_df = pd.read_csv(metrics_file)
    assert "label" in saved_metrics_df.columns
    assert "auc_roc" in saved_metrics_df.columns
    assert "accuracy" in saved_metrics_df.columns
    assert saved_metrics_df["auc_roc"].tolist() == auc_roc_series.tolist()
    assert saved_metrics_df["accuracy"].tolist() == accuracy_series.tolist()
