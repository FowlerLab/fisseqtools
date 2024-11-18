import pathlib

import numpy as np
import pandas as pd
import pytest

from fisseqtools.utils import (
    filter_labels,
    split_data,
    generate_splits,
    compute_pca,
    get_pca,
)


def test_filter_labels():
    data = pd.DataFrame(
        {
            "label": ["A", "A", "A", "B", "B", "C", "C", "C", "C", "D", "D", "F"],
            "value": range(12),
        }
    )

    filtered_data = filter_labels(
        data, label_col="label", frequency_cutoff=2, random_state=42
    )
    assert all(filtered_data["label"].value_counts() == 2)
    assert set(filtered_data["label"]) == {"A", "B", "C", "D"}
    assert len(filtered_data) == 8


def test_split_data():
    data = pd.DataFrame({"label": ["A"] * 50 + ["B"] * 50, "value": range(100)})

    x_train, x_eval, x_test = split_data(data, "label")
    assert len(x_train) == 80
    assert len(x_eval) == 10
    assert len(x_test) == 10
    assert x_train["label"].value_counts().to_dict() == {"A": 40, "B": 40}
    assert x_eval["label"].value_counts().to_dict() == {"A": 5, "B": 5}
    assert x_test["label"].value_counts().to_dict() == {"A": 5, "B": 5}

    data = pd.DataFrame(
        {"label": ["A"] * 55 + ["B"] * 55 + ["C"] * 10, "value": range(120)}
    )
    x_train, x_eval, x_test = split_data(
        data,
        "label",
    )


def test_generate_splits(tmp_path):
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "output"
    output_path.mkdir()

    data = pd.DataFrame({"label": ["A"] * 50 + ["B"] * 50, "value": range(100)})
    data.to_csv(data_path, index=False)

    generate_splits(data_path, "label", output_path)
    assert (output_path / "data.train.csv").exists()
    assert (output_path / "data.val.csv").exists()
    assert (output_path / "data.test.csv").exists()

    x_train = pd.read_csv(output_path / "data.train.csv")
    x_eval = pd.read_csv(output_path / "data.val.csv")
    x_test = pd.read_csv(output_path / "data.test.csv")
    assert len(x_train) == 80
    assert len(x_eval) == 10
    assert len(x_test) == 10
    assert x_train["label"].value_counts().to_dict() == {"A": 40, "B": 40}
    assert x_eval["label"].value_counts().to_dict() == {"A": 5, "B": 5}
    assert x_test["label"].value_counts().to_dict() == {"A": 5, "B": 5}
    assert "index" in x_train.columns
    assert "index" in x_eval.columns
    assert "index" in x_test.columns

    # Test with frequency cutoff
    data = pd.DataFrame(
        {"label": ["A"] * 55 + ["B"] * 55 + ["C"] * 20, "value": range(130)}
    )
    data.to_csv(data_path, index=False)

    generate_splits(data_path, "label", output_path, frequency_cutoff=50)
    assert (output_path / "data.train.csv").exists()
    assert (output_path / "data.val.csv").exists()
    assert (output_path / "data.test.csv").exists()

    x_train = pd.read_csv(output_path / "data.train.csv")
    x_eval = pd.read_csv(output_path / "data.val.csv")
    x_test = pd.read_csv(output_path / "data.test.csv")
    assert len(x_train) == 80
    assert len(x_eval) == 10
    assert len(x_test) == 10
    assert x_train["label"].value_counts().to_dict() == {"A": 40, "B": 40}
    assert x_eval["label"].value_counts().to_dict() == {"A": 5, "B": 5}
    assert x_test["label"].value_counts().to_dict() == {"A": 5, "B": 5}
    assert "index" in x_train.columns
    assert "index" in x_eval.columns
    assert "index" in x_test.columns


def test_compute_pca():
    features = np.random.rand(100, 10)
    reduced_features, max_error, min_error, median_error = compute_pca(features, 5)
    assert reduced_features.shape == (100, 5)
    assert 0 <= max_error <= 1
    assert 0 <= min_error <= 1
    assert 0 <= median_error <= 1
    assert min_error <= median_error <= max_error


def test_get_pca(tmp_path):
    features_path = tmp_path / "features.npy"
    pca_path = tmp_path / "reduced_features.npy"
    features = np.random.rand(100, 10)

    np.save(features_path, features)
    get_pca(features_path, n_components=5, pca_path=pca_path)
    assert pca_path.exists()

    reduced_features = np.load(pca_path)
    assert reduced_features.shape == (100, 5)
