"""Conduct hyperparameter search using successive halving"""

import concurrent.futures
import functools
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.model_selection


def sample_wt_sms(data_df: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample n wild type and single missense rows of data_df"""
    wt_indices = data_df[data_df["Variant_Class"] == "WT"].index.to_numpy()
    ms_indices = data_df[data_df["Variant_Class"] == "Single Missense"].index.to_numpy()
    select_wt_indices = np.random.choice(wt_indices, size=n, replace=False)
    select_ms_indices = np.random.choice(ms_indices, size=n, replace=False)

    return data_df.iloc[select_wt_indices], data_df.iloc[select_ms_indices]


def get_train_test_split(
    wt_df: pd.DataFrame,
    ms_df: pd.DataFrame,
    embeddings: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get training X and Y train and test arrays"""
    embedding_indices = np.concat((wt_df.index.to_numpy(), ms_df.index.to_numpy()))
    sample_embeddings = embeddings[embedding_indices]
    sample_labels = np.zeros(len(sample_embeddings), dtype=bool)
    sample_labels[: len(wt_df)] = True

    return sklearn.model_selection.train_test_split(
        sample_embeddings, sample_labels, test_size=test_size, stratify=sample_labels
    )


def test_hyperparams(
    classifier_hyperparams: Dict[str, Any],
    classifier_type: Type[sklearn.base.BaseEstimator | sklearn.base.ClassifierMixin],
    xtrain_xtest_ytrain_ytest: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> float:
    x_train, x_test, y_train, y_test = xtrain_xtest_ytrain_ytest
    classifier = classifier_type(**classifier_hyperparams).fit(x_train, y_train)
    return classifier.score(x_test, y_test), classifier_hyperparams

def successive_halving(
    start_dset_size: int,
    data_df: pd.DataFrame,
    embeddings: np.ndarray,
    classifier_type: Type[sklearn.base.BaseEstimator | sklearn.base.ClassifierMixin],
    hyperparams_list: List[Dict[int, Any]],
    num_threads: int = 1,
) -> Dict[Any, Any]:
    results_dict = dict()
    curr_hyperparams = hyperparams_list.copy()
    curr_round = 1

    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        while True:
            wt_df, ms_df = sample_wt_sms(data_df, start_dset_size)
            xtrain_xtest_ytrain_ytest = get_train_test_split(wt_df, ms_df, embeddings)
            test_fun = functools.partial(
                test_hyperparams,
                classifier_type=classifier_type,
                xtrain_xtest_ytrain_ytest=xtrain_xtest_ytrain_ytest,
            )

            hyperparam_scores = list(executor.map(test_fun, curr_hyperparams))
            hyperparam_scores.sort(reverse=True)
            results_dict[curr_round] = [
                params | {"accuracy": accuracy}
                for accuracy, params in hyperparam_scores
            ]
            
            if len(curr_hyperparams) <= 1:
                break

            curr_hyperparams = [params for _, params in hyperparam_scores][
                : len(hyperparam_scores) // 2
            ]
            curr_round += 1
            start_dset_size *= 2

    return results_dict
