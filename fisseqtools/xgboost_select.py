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


def get_train_val_split(
    val_split: float,
    train_data_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = math.ceil(1 / val_split)
    geno_counts = train_data_df["geno"].value_counts()
    train_data_df = train_data_df[
        train_data_df["geno"].isin(geno_counts[geno_counts >= n_samples].index)
    ]
    embeddings = embeddings[train_data_df["embedding_index"].to_numpy()]
    label_encoder = sklearn.preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(train_data_df["geno"])
    x_train, x_eval, y_train, y_eval = sklearn.model_selection.train_test_split(
        embeddings, labels, test_size=val_split, stratify=labels
    )
    
    shuffle_split = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=min(0.5, 3 * val_split)
    )
    sample_index, _ = next(shuffle_split.split(x_train, y_train))
    x_train_sample = x_train[sample_index]
    y_train_sample = y_train[sample_index]
    
    return x_train_sample, x_eval, y_train_sample, y_eval


def search_hyperparams(
    train_data_df_path: os.PathLike,
    embeddings_pkl_path: os.PathLike,
    output_path: os.PathLike,
    val_split: Optional[float] = 0.1,
    n_threads: Optional[int] = 1,
) -> None:
    train_data_df = pd.read_csv(train_data_df_path)
    with open(embeddings_pkl_path, "rb") as f:
        embeddings = pickle.load(f)

    x_train, x_eval, y_train, y_eval = get_train_val_split(
        val_split, train_data_df, embeddings
    )
    xgb_classifier = xgboost.XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        n_estimators=4096,
        early_stopping_rounds=16,
    )

    validation_fold = [-1] * len(x_train) + [0] * len(x_eval)
    train_val_split = sklearn.model_selection.PredefinedSplit(validation_fold)
    x_combined = np.vstack((x_train, x_eval))
    y_combined = np.concatenate((y_train, y_eval))
    sample_weights = sklearn.utils.class_weight.compute_sample_weight(
        class_weight="balanced", y=y_train
    )

    param_dist = {
        "max_depth": np.arange(1, 17, dtype=int),
        "learning_rate": 10 ** np.arange(-3, 0, dtype=float),
        "reg_alpha": 10 ** np.arange(-3, 0, dtype=float),
        "reg_lambda": 10 ** np.arange(-3, 0, dtype=float),
    }

    halving_random_search = sklearn.model_selection.RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=param_dist,
        n_iter=64,
        verbose=3,
        cv=train_val_split,
        n_jobs=n_threads,
    )

    halving_random_search.fit(
        x_combined,
        y_combined,
        eval_set=[(x_eval, y_eval)],
        sample_weight=sample_weights,
    )

    with open(pathlib.Path(output_path) / "best_params.json", "w") as f:
        json.dump(halving_random_search.best_params_, f)


def train_xgboost(
    train_data_df_path: os.PathLike,
    eval_data_df_path: os.PathLike,
    embeddings_pkl_path: os.PathLike,
    output_dir: os.PathLike,
) -> None:

if __name__ == "__main__":
    fire.Fire({"search": search_hyperparams})
