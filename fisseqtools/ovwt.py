import functools
import itertools
import json
import pathlib
import pickle
from os import PathLike
from typing import Callable, Dict, List, Optional, Tuple

import fire
import numpy as np
import pandas as pd
import shap
import sklearn
import sklearn.metrics
import sklearn.utils
import tqdm
import xgboost as xgb

TrainFun = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray | None]],
    sklearn.base.BaseEstimator,
]


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    """
    Trains an XGBoost classifier on the provided training data.

    Args:
        x_train (np.ndarray):
            Feature matrix for training data.
        y_train (np.ndarray):
            Labels for training data.
        x_eval (np.ndarray):
            Feature matrix for evaluation data.
        y_eval (np.ndarray):
            Labels for evaluation data.
        sample_weight (Optional[np.ndarray | None], optional):
            Sample weights for training, default is None.

    Returns:
        sklearn.base.BaseEstimator:
            The trained XGBoost classifier.
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        colsample_bynode=0.7,
        subsample=0.5,
        early_stopping_rounds=5,
        n_estimators=100,
        eval_metric="auc",
    ).fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_eval, y_eval)],
        sample_weight=sample_weight,
        verbose=True,
    )


def get_mask_features(
    target_column: str,
    feature_columns: List[str],
    wt_key: str,
    curr_split: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a mask for the wild-type entries and extracts the feature matrix.

    Args:
        target_column (str):
            The name of the target column.
        feature_columns (List[str]):
            List of feature column names.
        wt_key (str):
            The wild-type key to identify the wild-type entries.
        curr_split (pd.DataFrame):
            The dataset split to be processed.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the wild-type mask (wt_mask) and the feature
            matrix.
    """
    wt_mask = (curr_split[target_column] == wt_key).to_numpy(dtype=bool)
    feature_matrix = curr_split[feature_columns].to_numpy()
    return wt_mask, feature_matrix


def get_train_data_labels(
    wt_mask: np.ndarray,
    variant_mask: np.ndarray,
    feature_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the feature data and labels for wild-type and variant samples from
    the feature matrix.

    Args:
        wt_mask (np.ndarray):
            Mask for the wild-type entries.
        variant_mask (np.ndarray):
            Mask for the variant entries.
        feature_matrix (np.ndarray):
            The matrix containing feature values.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the combined feature matrix and the
            corresponding labels.
    """
    wt_features = feature_matrix[wt_mask]
    variant_features = feature_matrix[variant_mask]
    combined_features = np.vstack((wt_features, variant_features))
    labels = np.zeros(len(combined_features), dtype=bool)
    labels[len(wt_features) :] = True
    return combined_features, labels


def get_metrics(
    model: sklearn.base.ClassifierMixin,
    features: np.ndarray,
    labels: np.ndarray,
    dataset_name: Optional[str] = "",
    verbose: Optional[str] = True,
) -> Tuple[float, float]:
    """
    Computes and returns the ROC AUC and accuracy.

    Args:
        model (sklearn.base.ClassifierMixin):
            The trained classifier.
        features (np.ndarray):
            Feature matrix for the dataset.
        labels (np.ndarray):
            True labels for the dataset.
        dataset_name (Optional[str], optional):
            The name of the dataset, default is an empty string.
        verbose (Optional[str], optional):
            Whether to print the metrics, default is True.

    Returns:
        Tuple[float, float]:
            The ROC AUC and accuracy of the classifier on the dataset.
    """
    y_prob = model.predict_proba(features)[:, 1].flatten()
    y_pred = y_prob >= 0.5
    roc_auc = sklearn.metrics.roc_auc_score(labels, y_prob)
    accuracy = sklearn.metrics.accuracy_score(labels, y_pred)

    if verbose:
        if dataset_name != "":
            dataset_name += " "

        print(f"{dataset_name}ROC AUC: {roc_auc:.2f}", flush=True)
        print(f"{dataset_name}Accuracy: {accuracy:.2f}", flush=True)

    return roc_auc, accuracy


def train_ovwt(
    train_fun: TrainFun,
    train_split: pd.DataFrame,
    eval_one_split: pd.DataFrame,
    eval_two_split: pd.DataFrame,
    meta_data: Dict[str, str | List[str]],
    wt_key: Optional[str] = "WT",
) -> Tuple[Dict[str, sklearn.base.BaseEstimator], pd.DataFrame]:
    """
    Trains models for different variants of a target column.

    Args:
        train_fun (TrainFun):
            Function used to train the model.
        train_split (pd.DataFrame):
            DataFrame containing the training data.
        eval_one_split (pd.DataFrame):
            DataFrame containing the first evaluation split.
        eval_two_split (pd.DataFrame):
            DataFrame containing the second evaluation split.
        meta_data (Dict[str, str | List[str]]):
            Metadata containing column names for target and features.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".

    Returns:
        Tuple[Dict[str, sklearn.base.BaseEstimator], pd.DataFrame]:
            A tuple containing the trained models and the metrics DataFrame.
    """
    target_column: str = meta_data["target_column"]
    feature_columns: List[str] = meta_data["feature_columns"]

    get_features = functools.partial(
        get_mask_features, target_column, feature_columns, wt_key
    )
    train_wt_mask, train_features = get_features(train_split)
    eval_one_wt_mask, eval_one_features = get_features(eval_one_split)
    eval_two_wt_mask, eval_two_features = get_features(eval_two_split)

    datasets = ["eval_two", "eval_one", "train"]
    stats = ["roc_auc", "accuracy"]
    accuracy_roc = {
        f"{dataset}_{stat}": list()
        for dataset, stat in itertools.product(datasets, stats)
    }
    accuracy_roc[target_column] = list()
    models = dict()

    for curr_variant in train_split[target_column].unique():
        if curr_variant == wt_key:
            print("WT Key, Skipping\n", flush=True)
            continue

        print(f"Training classifier {curr_variant}", flush=True)

        # Get data and labels
        get_var_mask = lambda split: (split[target_column] == curr_variant).to_numpy()
        curr_train_features, curr_train_labels = get_train_data_labels(
            train_wt_mask, get_var_mask(train_split), train_features
        )
        curr_eval_one_features, curr_eval_one_labels = get_train_data_labels(
            eval_one_wt_mask, get_var_mask(eval_one_split), eval_one_features
        )
        curr_eval_two_features, curr_eval_two_labels = get_train_data_labels(
            eval_two_wt_mask, get_var_mask(eval_two_split), eval_two_features
        )
        weights = sklearn.utils.compute_sample_weight("balanced", curr_train_labels)

        model = train_fun(
            curr_train_features,
            curr_train_labels,
            curr_eval_one_features,
            curr_eval_one_labels,
            weights,
        )

        curr_datasets = [
            ("Train", curr_train_features, curr_train_labels),
            ("Eval One", curr_eval_one_features, curr_eval_one_labels),
            ("Eval Two", curr_eval_two_features, curr_eval_two_labels),
        ]

        for name, features, labels in curr_datasets:
            roc_auc, accuracy = get_metrics(model, features, labels, dataset_name=name)
            name = name.lower().replace(" ", "_")
            accuracy_roc[name + "_roc_auc"].append(roc_auc)
            accuracy_roc[name + "_accuracy"].append(accuracy)

        accuracy_roc[target_column].append(curr_variant)
        models[curr_variant] = model
        print("", flush=True)

    return models, pd.DataFrame(accuracy_roc)


def get_shap_values(
    dataset: pd.DataFrame,
    models: Dict[str, sklearn.base.BaseEstimator],
    meta_data: Dict[str, str | List[str]],
    wt_key: Optional[str] = "WT",
    dset_name: Optional[str | None] = None,
) -> pd.DataFrame:
    """
    Computes SHAP values for the given dataset over all variants.

    Args:
        dataset (pd.DataFrame):
            The dataset for which to compute SHAP values.
        models (Dict[str, sklearn.base.BaseEstimator]):
            The models to use for SHAP value computation.
        meta_data (Dict[str, str | List[str]]):
            Metadata containing column names for target and features.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
        dset_name (Optional[str | None], optional):
            The name of the dataset, used for progress display, default is None.

    Returns:
        pd.DataFrame:
            A DataFrame containing SHAP values for each sample and feature.
    """
    target_column: str = meta_data["target_column"]
    feature_columns: List[str] = meta_data["feature_columns"]
    dataset = dataset[dataset[target_column] != wt_key]
    all_features = dataset[feature_columns].to_numpy()

    # Initialize shap value dataframe
    filtered_dataset = dataset[dataset[target_column] != wt_key]
    shap_columns = [target_column] + ["p_is_var"] + feature_columns
    shap_df = pd.DataFrame(
        [[None] * len(shap_columns)] * len(dataset), columns=shap_columns
    )
    shap_df.loc[:, target_column] = filtered_dataset[target_column].to_numpy()

    pbar_desc = (
        None if dset_name is None else f"Computing Shap Values Over: {dset_name}"
    )
    for curr_variant in tqdm.tqdm(shap_df[target_column].unique(), desc=pbar_desc):
        curr_model = models[curr_variant]
        variant_mask = (shap_df[target_column] == curr_variant).to_numpy()
        curr_features = all_features[variant_mask]

        curr_explainer = shap.TreeExplainer(curr_model)
        curr_shap_vals = curr_explainer.shap_values(curr_features)[:, :, 1]
        shap_df.loc[variant_mask, feature_columns] = curr_shap_vals

        curr_predict_probas = curr_model.predict_proba(curr_features)[:, 1]
        shap_df["p_is_var"][variant_mask] = curr_predict_probas

    return shap_df


def ovwt(
    train_fun: TrainFun,
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    eval_two_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    wt_key: Optional[str] = "WT",
) -> None:
    """
    Orchestrates the training, evaluation, and SHAP value computation.

    Args:
        train_fun (TrainFun):
            Function used to train the model.
        train_data_path (PathLike):
            Path to the training data file.
        eval_one_data_path (PathLike):
            Path to the first evaluation data file.
        eval_two_data_path (PathLike):
            Path to the second evaluation data file.
        meta_data_json_path (PathLike):
            Path to the metadata JSON file.
        output_dir (PathLike):
            Directory where results and models are saved.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
    """
    train_split = pd.read_parquet(train_data_path)
    eval_one_split = pd.read_parquet(eval_one_data_path)
    eval_two_split = pd.read_parquet(eval_two_data_path)

    with open(meta_data_json_path) as f:
        meta_data = json.load(f)

    output_dir = pathlib.Path(output_dir)

    models, accuracy_roc = train_ovwt(
        train_fun, train_split, eval_one_split, eval_two_split, meta_data, wt_key
    )

    accuracy_roc.to_csv(output_dir / "train_results.csv")
    with open(output_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    shap_targets = [
        ("train", train_split),
        ("eval_one", eval_one_split),
        ("eval_two", eval_two_split),
    ]

    for name, split in shap_targets:
        curr_shap_df = get_shap_values(split, models, meta_data, wt_key, name)
        curr_shap_df.to_parquet(output_dir / f"{name}_shap.parquet")


if __name__ == "__main__":
    fire.Fire({"xgb": functools.partial(ovwt, train_xgboost)})
