import math
import os
import pathlib
import pickle
import warnings
from typing import Optional, Tuple

import fire
import json
import numpy as np
import pandas as pd
import sklearn.experimental.enable_halving_search_cv
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.utils.class_weight
import xgboost

from .hyperparam_search import sample_wt_sms, get_train_test_split


def get_even_training_data(
    n_samples: int,
    train_data_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    geno_counts = train_data_df["geno"].value_counts()
    train_data_df = train_data_df[
        train_data_df["geno"].isin(geno_counts[geno_counts >= n_samples].index)
    ]

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(train_data_df["geno"])
    labels = label_encoder.transform(train_data_df["geno"])

    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples * len(np.unique(labels))
    )
    data_idx, _ = next(splitter.split(embeddings, labels))
    return embeddings[data_idx], labels[data_idx]


def search_hyperparams(
    train_data_df_path: os.PathLike,
    embeddings_pkl_path: os.PathLike,
    output_path: os.PathLike,
    select_top_k: Optional[int | None] = None,
    mutual_info_npy: Optional[os.PathLike | None] = None,
    n_geno_samples: Optional[int] = 220,
    n_threads: Optional[int] = 1,
) -> None:
    train_data_df = pd.read_csv(train_data_df_path)
    with open(embeddings_pkl_path, "rb") as f:
        embeddings = pickle.load(f)
    embeddings = embeddings[train_data_df["embedding_index"].to_numpy()]

    if select_top_k is not None:
        if mutual_info_npy is None:
            warnings.warn(
                "A Mutual Info npy file must be provided to select top k features."
                " Top k selection will not be performed.",
                RuntimeWarning,
            )
        else:
            mutual_info = np.load(mutual_info_npy)
            top_k_embeddings_idx = mutual_info.argsort()[::-1]
            top_k_embeddings_idx = top_k_embeddings_idx[:select_top_k]
            embeddings = embeddings[:, top_k_embeddings_idx]

    x_data, y_data = get_even_training_data(n_geno_samples, train_data_df, embeddings)
    xgb_classifier = xgboost.XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_data)),
        eval_metric="mlogloss",
    )
    x_train, x_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        x_data, y_data, test_size=0.1, stratify=y_data
    )

    param_dist = {
        "max_depth": np.arange(1, 17, dtype=int),
        "learning_rate": 10 ** np.arange(-3, 0, dtype=float),
        "reg_alpha": 10 ** np.arange(-3, 0, dtype=float),
        "reg_lambda": 10 ** np.arange(-3, 0, dtype=float),
    }

    halving_random_search = sklearn.model_selection.HalvingRandomSearchCV(
        estimator=xgb_classifier,
        param_distributions=param_dist,
        factor=2,
        resource="n_estimators",
        max_resources=2048,
        min_resources=64,
        verbose=3,
        cv=5,
        n_jobs=n_threads,
    )

    halving_random_search.fit(
        x_train,
        y_train,
        eval_set=([x_eval, y_eval]),
        early_stopping_rounds=16,
    )

    with open(pathlib.Path(output_path) / "best_params.json", "w") as f:
        json.dump(halving_random_search.best_params_, f)

if __name__ == "__main__":
    fire.Fire({"search": search_hyperparams})
