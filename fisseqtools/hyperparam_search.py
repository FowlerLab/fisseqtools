"""Conduct hyperparameter search using successive halving"""

import concurrent.futures
import functools
import math
import os
import pickle
import pathlib
from typing import Any, Dict, List, Tuple, Type, TextIO

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.model_selection
import sklearn.ensemble
import tqdm


def sample_wt_sms(
    data_df: pd.DataFrame,
    n: int,
    wildtype_class: str = "WT",
    mutant_classes: List[str] = ["Single Missense", "Multiple", "Nonsense"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample n wild type and n mutant rows of data_df"""
    wt_indices = data_df[data_df["Variant_Class"] == wildtype_class].index.to_numpy()
    ms_indices = data_df[data_df["Variant_Class"].isin(mutant_classes)].index.to_numpy()
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
    embedding_indices = np.concat(
        (wt_df["embedding_index"].to_numpy(), ms_df["embedding_index"].to_numpy())
    )
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

    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor, tqdm.tqdm(
        total=math.ceil(math.log2(len(curr_hyperparams)))
    ) as main_pbar:
        while True:
            wt_df, ms_df = sample_wt_sms(data_df, start_dset_size)
            xtrain_xtest_ytrain_ytest = get_train_test_split(wt_df, ms_df, embeddings)
            test_fun = functools.partial(
                test_hyperparams,
                classifier_type=classifier_type,
                xtrain_xtest_ytrain_ytest=xtrain_xtest_ytrain_ytest,
            )

            futures = [executor.submit(test_fun, params) for params in curr_hyperparams]
            hyperparam_scores = [
                future.result()
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    leave=False,
                )
            ]
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
            main_pbar.update()

    return results_dict


def search_gradient_boost_hyperparams(
    data_df_path: os.PathLike,
    embeddings_pkl_path: os.PathLike,
    results_path: os.PathLike,
    num_threads: int = 1,
    max_depth: int = 32,
    num_estimators: int = 128,
    start_dset_size: int = 1024,
) -> None:
    data_df = pd.read_csv(data_df_path)
    with open(embeddings_pkl_path, "rb") as f:
        embeddings = pickle.load(f)

    hyperparams = [
        {"max_depth": i, "n_estimators": num_estimators}
        for i in range(1, max_depth + 1)
    ]

    results = successive_halving(
        start_dset_size,
        data_df,
        embeddings,
        sklearn.ensemble.GradientBoostingClassifier,
        hyperparams,
        num_threads=num_threads,
    )


def split_pheno_data(data_csv_path: TextIO, strat_col: str = "geno") -> None:
    data_df = pd.read_csv(data_csv_path)
    data_df["embedding_index"] = data_df.index
    strat_labels = data_df[strat_col]

    x_train, x_test_val, y_train, y_test_val = sklearn.model_selection.train_test_split(
        data_df, strat_labels, train_size=0.8, stratify=strat_labels
    )
    x_test, x_val, y_test, y_val = sklearn.model_selection.train_test_split(
        x_test_val, y_test_val, train_size=0.5, stratify=y_test_val
    )

    data_csv_path = pathlib.Path(data_csv_path)
    x_train.to_csv(data_csv_path.with_suffix(".train.csv"))
    x_val.to_csv(data_csv_path.with_suffix(".val.csv"))
    x_test.to_csv(data_csv_path.with_suffix(".test.csv"))
