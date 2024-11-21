import os
import pathlib
from typing import Tuple, Optional, Iterable

import fire
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.model_selection


def filter_labels(
    data: pd.Series,
    label_col: str,
    frequency_cutoff: int,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    label_counts = data[label_col].value_counts()
    freq_mask = data[label_col].map(label_counts) >= frequency_cutoff
    selected_rows = data[freq_mask]
    selected_rows = selected_rows.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(n=frequency_cutoff, random_state=random_state)
    )

    return selected_rows


def split_data(
    data_df: pd.DataFrame, label_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_temp = sklearn.model_selection.train_test_split(
        data_df, test_size=0.2, stratify=data_df[label_col]
    )
    x_eval, x_test = sklearn.model_selection.train_test_split(
        x_temp, test_size=0.5, stratify=x_temp[label_col]
    )
    return x_train, x_eval, x_test


def generate_splits(
    data_df_path: os.PathLike,
    label_col: str,
    output_path: os.PathLike,
    frequency_cutoff: Optional[int | None] = None,
) -> None:
    data_df_path = pathlib.Path(data_df_path)
    data_df = pd.read_csv(data_df_path)
    data_df["index"] = np.arange(len(data_df))
    if frequency_cutoff is not None:
        data_df = filter_labels(data_df, label_col, frequency_cutoff)

    train_df, eval_df, test_df = split_data(data_df, label_col)
    output_file_prefix = pathlib.Path(output_path) / data_df_path.stem
    train_df.to_csv(output_file_prefix.with_suffix(".train.csv"))
    eval_df.to_csv(output_file_prefix.with_suffix(".val.csv"))
    test_df.to_csv(output_file_prefix.with_suffix(".test.csv"))


def compute_pca(
    features: np.ndarray, n_components: int
) -> Tuple[np.ndarray, float, float, float]:
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca = pca.fit(features)
    reduced_features = pca.transform(features)
    reconstructed_features = pca.inverse_transform(reduced_features)
    reconstruction_diff = features - reconstructed_features
    reconstruction_error = np.linalg.norm(reconstruction_diff, axis=0) / np.linalg.norm(
        features, axis=0
    )

    return (
        reduced_features,
        np.max(reconstruction_error),
        np.min(reconstruction_error),
        np.median(reconstruction_error),
    )


def get_pca(
    features_path: os.PathLike, n_components: int, pca_path: os.PathLike
) -> None:
    features = np.load(features_path)
    reduced_features, max_error, min_error, median_error = compute_pca(
        features, n_components
    )
    print("Feature Reduction Reconstruction Error:")
    print(f"    Max reconstruction error: {max_error:.2%}")
    print(f"    Min reconstruction error: {min_error:.2%}")
    print(f"    Median reconstruction error: {median_error:.2%}")
    np.save(pca_path, reduced_features)


def save_metrics(
    data_df: pd.DataFrame,
    auc_roc_series: pd.Series,
    accuracy_series: pd.Series,
    select_key: str,
    output_path: pathlib.Path,
    label_true: Iterable[str],
    label_pred: Iterable[str],
) -> None:
    metrics_df = pd.DataFrame({"label": data_df[select_key].unique()})
    metrics_df["auc_roc"] = metrics_df["label"].map(auc_roc_series)
    metrics_df["accuracy"] = metrics_df["label"].map(accuracy_series)
    metrics_df.to_csv(output_path / "metrics.csv", index=False)
    pd.DataFrame({"true_label": label_true, "label_predicted": label_pred}).to_csv(
        output_path / "predictions.csv"
    )


if __name__ == "__main__":
    fire.Fire({"splits": generate_splits, "pca": get_pca})
