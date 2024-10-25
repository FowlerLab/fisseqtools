import concurrent.futures
import csv
import enum
import functools
import os
import pathlib
import pickle
from typing import List, Tuple, Dict, Any

import fire
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import tqdm


class VarTag(enum.Enum):
    TARGET_VAR = 0
    WILD_TYPE = 1
    NON_TARGET_MUTANT = 2


def prep_train_data(
    variant: str,
    data_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    wildtype_class: str = "WT"
    mutant_classes: List[str] = ["Single Missense", "Multiple", "Nonsense"]

    target_df = data_df[data_df["geno"] == variant]
    n_target = len(target_df)
    non_target_df = data_df[data_df["geno"] != variant]
    wt_df = non_target_df[non_target_df["Variant_Class"] == wildtype_class].sample(
        n=n_target // 2
    )
    mutant_df = non_target_df[
        non_target_df["Variant_Class"].isin(mutant_classes)
    ].sample(n=n_target // 2)

    embedding_indices = np.concat(
        (
            target_df["embedding_index"].to_numpy(),
            wt_df["embedding_index"].to_numpy(),
            mutant_df["embedding_index"].to_numpy(),
        )
    )

    sample_embeddings = embeddings[embedding_indices]
    sample_labels = np.concat(
        (
            np.repeat(VarTag.TARGET_VAR.value, len(target_df)),
            np.repeat(VarTag.WILD_TYPE.value, len(wt_df)),
            np.repeat(VarTag.NON_TARGET_MUTANT.value, len(mutant_df)),
        )
    )

    return sample_embeddings, sample_labels


def train_model(
    classifier_hyperparams: Dict[str, Any],
    data_df: pd.DataFrame,
    embeddings: pd.DataFrame,
    target_variant: str,
) -> Tuple[
    sklearn.base.BaseEstimator | sklearn.base.ClassifierMixin, float, float, float
]:
    sample_embeddings, sample_labels = prep_train_data(
        target_variant, data_df, embeddings
    )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        sample_embeddings, sample_labels, test_size=0.1, stratify=sample_labels
    )

    y_test_wt = y_test[y_test == VarTag.WILD_TYPE.value]
    x_test_wt = x_test[y_test == VarTag.WILD_TYPE.value]
    y_test_mutant = y_test[y_test == VarTag.NON_TARGET_MUTANT.value]
    x_test_mutant = x_test[y_test == VarTag.NON_TARGET_MUTANT.value]

    # Sample target variant samples form test set for mutant and wt sets
    target_test_idx = np.arange(len(y_test))[y_test == VarTag.TARGET_VAR.value]
    np.random.shuffle(target_test_idx)
    target_test_wt_idx = target_test_idx[: len(y_test_wt)]
    target_test_mutant_idx = target_test_idx[len(y_test_wt) :]

    y_test_wt = np.concat((y_test_wt, y_test[target_test_wt_idx]))
    x_test_wt = np.concat((x_test_wt, x_test[target_test_wt_idx]))
    y_test_mutant = np.concat((y_test_mutant, y_test[target_test_mutant_idx]))
    x_test_mutant = np.concat((x_test_mutant, x_test[target_test_mutant_idx]))

    for test_vec in [y_test, y_test_wt, y_test_mutant]:
        test_vec[:] = test_vec == VarTag.TARGET_VAR.value

    model = sklearn.ensemble.GradientBoostingClassifier(**classifier_hyperparams)
    model.fit(x_train, y_train)

    y_pred_test = model.predict_proba(x_test)[:, 1]
    y_pred_test_wt = model.predict_proba(x_test_wt)[:, 1]
    y_pred_test_mutant = model.predict_proba(x_test_mutant)[:, 1]

    test_auc = sklearn.metrics.roc_auc_score(y_test, y_pred_test)
    test_wt_auc = sklearn.metrics.roc_auc_score(y_test_wt, y_pred_test_wt)
    test_mutant_auc = sklearn.metrics.roc_auc_score(y_test_mutant, y_pred_test_mutant)

    return model, test_auc, test_wt_auc, test_mutant_auc


def train_models(
    data_df_path: os.PathLike,
    embeddings_pkl_path: os.PathLike,
    results_dir_path: os.PathLike,
    num_threads: int = 1,
    max_depth: int = 17,
    max_estimators: int = 15360,
    early_stop_iter: int = 16,
    min_cell_count: int = 2000,
) -> None:
    hparams = {
        "max_depth": max_depth,
        "n_estimators": max_estimators,
        "n_iter_no_change": early_stop_iter,
    }

    # Filter any variants that occur less than 2000 times
    data_df = pd.read_csv(data_df_path)
    geno_counts = data_df["geno"].value_counts()
    variant_df = data_df[
        data_df["geno"].isin(geno_counts[geno_counts >= min_cell_count].index)
    ]
    variants = variant_df["geno"].unique()

    with open(embeddings_pkl_path, "rb") as f:
        embeddings = pickle.load(f)

    results_dir_path = pathlib.Path(results_dir_path)
    classifiers_path = results_dir_path / "classifiers"
    classifiers_path.mkdir(exist_ok=True)
    train_fun = functools.partial(train_model, hparams, data_df, embeddings)

    with concurrent.futures.ProcessPoolExecutor(num_threads) as executor:
        futures = [executor.submit(train_fun, variant) for variant in variants]
        training_results = [
            (*future.result(), geno)
            for future, geno in zip(
                tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    desc="Training Models",
                    total=len(futures),
                ),
                variants,
            )
        ]

    with open(results_dir_path / "results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["classifier_fname", "geno", "test_auc", "wt_auc", "mutant_auc"],
        )
        writer.writeheader()

        for results_idx, (
            model,
            test_auc,
            test_wt_auc,
            test_mutant_auc,
            geno,
        ) in enumerate(tqdm.tqdm(training_results, desc="Processing Results"), 1):
            classifier_fname = f"gboost_{results_idx}.pkl"
            with open(classifiers_path / classifier_fname, "wb") as f:
                pickle.dump(model, f)

            writer.writerow(
                {
                    "classifier_fname": classifier_fname,
                    "geno": geno,
                    "test_auc": test_auc,
                    "wt_auc": test_wt_auc,
                    "mutant_auc": test_mutant_auc,
                }
            )


if __name__ == "__main__":
    fire.Fire({"select": train_models})
