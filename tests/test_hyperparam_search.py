import fisseqtools.hyperparam_search

import pathlib
import random
import unittest.mock

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


@pytest.fixture
def data_df():
    data_df = pd.DataFrame(
        {
            "Variant_Class": ["WT"] * 10
            + random.choices(["Single Missense", "Multiple", "Nonsense"], k=10),
            "Feature1": np.random.rand(20),
            "Feature2": np.random.rand(20),
        }
    )
    data_df["embedding_index"] = data_df.index
    return data_df


@pytest.fixture
def embeddings():
    return np.random.rand(20, 10)


@pytest.fixture
def mock_hyperparams_list():
    return [{"strategy": "most_frequent"}, {"strategy": "uniform"}]


def DummyClassifier(*args, **kwargs):
    dummy_classifier = unittest.mock.MagicMock()
    dummy_classifier.score = lambda *args, **kwargs: random.random()
    dummy_classifier.fit.return_value = dummy_classifier
    return dummy_classifier


def test_sample_wt_sms(data_df):
    wt_df, ms_df = fisseqtools.hyperparam_search.sample_wt_sms(data_df, 2)
    assert len(wt_df) == 2
    assert len(ms_df) == 2
    assert all(wt_df["Variant_Class"] == "WT")
    assert all(ms_df["Variant_Class"].isin(["Single Missense", "Multiple", "Nonsense"]))


def test_get_train_test_split(data_df, embeddings):
    wt_df, ms_df = fisseqtools.hyperparam_search.sample_wt_sms(data_df, 5)
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = fisseqtools.hyperparam_search.get_train_test_split(wt_df, ms_df, embeddings)

    assert x_train.shape[0] == 8  # 80% of 4 samples
    assert x_test.shape[0] == 2  # 20% of 4 samples
    assert y_train.shape == x_train.shape[:1]
    assert y_test.shape == x_test.shape[:1]


def test_test_hyperparams():
    x_train = np.array([[1, 0], [1, 0]])
    x_test = np.array([[1, 0], [1, 0]])
    y_train = np.array([[1], [0]])
    y_test = np.array([[1], [0]])
    hyperparams = {"strategy": "most_frequent"}

    result, returned_hyperparams = fisseqtools.hyperparam_search.test_hyperparams(
        classifier_hyperparams=hyperparams,
        classifier_type=DummyClassifier,
        xtrain_xtest_ytrain_ytest=(x_train, x_test, y_train, y_test),
    )

    assert isinstance(result, float)
    assert hyperparams == returned_hyperparams


def test_successive_halving(data_df, embeddings, mock_hyperparams_list):
    result = fisseqtools.hyperparam_search.successive_halving(
        start_dset_size=5,
        data_df=data_df,
        embeddings=embeddings,
        classifier_type=DummyClassifier,
        hyperparams_list=mock_hyperparams_list,
        num_threads=1,
    )

    assert isinstance(result, dict)
    assert len(result) == 2
    assert len(result[1]) == 2
    assert len(result[2]) == 1
    assert "accuracy" in result[1][0]


def test_split_pheno_data(tmpdir):
    csv_file_path = pathlib.Path(tmpdir.join("data.csv"))
    test_data = pd.DataFrame(
        {
            "geno": ["A", "B"] * 10,
            "Feature1": np.random.rand(20),
            "Feature2": np.random.rand(20),
        }
    )
    test_data.to_csv(csv_file_path)
    fisseqtools.hyperparam_search.split_pheno_data(csv_file_path)

    # Expected output file paths
    train_file_path = csv_file_path.with_suffix(".train.csv")
    val_file_path = csv_file_path.with_suffix(".val.csv")
    test_file_path = csv_file_path.with_suffix(".test.csv")

    assert train_file_path.exists()
    assert val_file_path.exists()
    assert test_file_path.exists()

    train_df = pd.read_csv(train_file_path)
    val_df = pd.read_csv(val_file_path)
    test_df = pd.read_csv(test_file_path)

    # Assert that the splits have the expected number of rows
    assert len(train_df) == 16
    assert len(val_df) == 2
    assert len(test_df) == 2

    assert "embedding_index" in train_df.columns
    assert "embedding_index" in val_df.columns
    assert "embedding_index" in test_df.columns

    assert train_df["geno"].value_counts()["A"] == 8
    assert train_df["geno"].value_counts()["B"] == 8
    assert val_df["geno"].value_counts()["A"] == 1
    assert val_df["geno"].value_counts()["B"] == 1
    assert test_df["geno"].value_counts()["A"] == 1
    assert test_df["geno"].value_counts()["B"] == 1
