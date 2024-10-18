import fisseqtools.hyperparam_search

import random
import unittest.mock

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin


@pytest.fixture
def data_df():
    return pd.DataFrame(
        {
            "Variant_Class": ["WT"] * 10
            + random.choices(["Single Missense", "Multiple", "Nonsense"], k=10),
            "Feature1": np.random.rand(20),
            "Feature2": np.random.rand(20),
        }
    )


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
