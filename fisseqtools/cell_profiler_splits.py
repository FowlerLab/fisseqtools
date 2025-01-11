import json
import math
import pathlib
from os import PathLike

import fire
import numpy as np
import pandas as pd
import sklearn.model_selection

random_state = np.random.RandomState(seed=42)


def filter_stratify_replicates(
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    """
    Filters and balances two DataFrames (r1_df and r2_df) by stratifying based
    on common `aaChanges` values. Ensures the counts for each `aaChanges`
    value are balanced between the two replicates, with a minimum threshold
    `min_count`.

    Parameters:
        r1_df (pd.DataFrame):
            First replicate DataFrame containing an `aaChanges` column.
        r2_df (pd.DataFrame):
            Second replicate DataFrame containing an `aaChanges` column.
        min_count (int):
            Minimum combined count for `aaChanges` to be included.

    Returns:
        pd.DataFrame
    """
    r1_df = r1_df.reset_index(drop=True)
    r2_df = r2_df.reset_index(drop=True)

    print(r1_df.index.min(), r1_df.index.max())
    print(r2_df.index.min(), r2_df.index.max())

    r1_aa_counts = r1_df["aaChanges"].value_counts()
    r2_aa_counts = r2_df["aaChanges"].value_counts()
    r1_aa_counts = r1_aa_counts[r1_aa_counts >= math.ceil(min_count / 2)]
    r2_aa_counts = r2_aa_counts[r2_aa_counts >= math.ceil(min_count / 2)]
    common_aas = r1_aa_counts.index.intersection(r2_aa_counts.index)

    r1_mask = np.zeros(len(r1_df), dtype=bool)
    r2_mask = np.zeros(len(r2_df), dtype=bool)
    r1_data_mask = (r1_df, r1_mask)
    r2_data_mask = (r2_df, r2_mask)

    for cur_aa in common_aas:
        if r1_aa_counts[cur_aa] > r2_aa_counts[cur_aa]:
            small_df, small_mask = r2_data_mask
            large_df, large_mask = r1_data_mask
            sample_n = r2_aa_counts[cur_aa]
        else:
            small_df, small_mask = r1_data_mask
            large_df, large_mask = r2_data_mask
            sample_n = r1_aa_counts[cur_aa]

        small_rows = small_df[small_df["aaChanges"] == cur_aa].index.to_numpy()
        large_rows = large_df[large_df["aaChanges"] == cur_aa].index.to_numpy()
        large_rows = random_state.choice(large_rows, size=sample_n, replace=False)
        small_mask[small_rows] = True
        large_mask[large_rows] = True

    return pd.concat((r1_df[r1_mask], r2_df[r2_mask]))


def filter_no_stratify(
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    min_count: int,
) -> pd.DataFrame:
    """
    Filters two DataFrames (r1_df and r2_df) based on common `aaChanges` values
    that meet a minimum combined count threshold `min_count`.

    Parameters:
        r1_df (pd.DataFrame):
            First replicate DataFrame containing an `aaChanges` column.
        r2_df (pd.DataFrame):
            Second replicate DataFrame containing an `aaChanges` column.
        min_count (int):
            Minimum combined count for `aaChanges` to be included.

    Returns:
        pd.DataFrame
    """
    r1_df = r1_df.reset_index(drop=True)
    r2_df = r2_df.reset_index(drop=True)

    r1_aa_counts = r1_df["aaChanges"].value_counts()
    r2_aa_counts = r2_df["aaChanges"].value_counts()
    common_aas = r1_aa_counts.index.intersection(r2_aa_counts.index)
    combined_aa_counts = r1_aa_counts[common_aas] + r2_aa_counts[common_aas]
    combined_aa_counts = combined_aa_counts[combined_aa_counts >= min_count]

    r1_df = r1_df[r1_df["aaChanges"].isin(combined_aa_counts.index)]
    r2_df = r2_df[r2_df["aaChanges"].isin(combined_aa_counts.index)]
    return pd.concat((r1_df, r2_df))


def get_splits(
    r1_parquet_path: PathLike,
    r2_parquet_path: PathLike,
    output_dir: PathLike,
    stratify_replicates: bool = False,
    min_count: int = 0,
) -> None:
    """
    Reads two Parquet files, filters and processes the data based on
    `aaChanges`, and splits it into training, evaluation, and test datasets.
    Outputs the splits as Parquet files and creates a JSON file with feature
    metadata.

    Parameters:
        r1_parquet_path (PathLike):
            Path to the first Parquet file.
        r2_parquet_path (PathLike):
            Path to the second Parquet file.
        output_dir (PathLike):
            Directory to save the output files.
        stratify_replicates (bool, optional):
            If True, stratifies the filtering process to balance replicates.
        min_count (int, optional):
            Minimum combined count for `aaChanges` to be included.
    """
    output_dir = pathlib.Path(output_dir)
    r1_df = pd.read_parquet(r1_parquet_path)
    r2_df = pd.read_parquet(r2_parquet_path)
    r1_df["replicate"] = 1
    r2_df["replicate"] = 2

    filter_fun = (
        filter_no_stratify if not stratify_replicates else filter_stratify_replicates
    )
    combined_df = filter_fun(r1_df, r2_df, min_count)

    train_split, remainder = sklearn.model_selection.train_test_split(
        combined_df,
        test_size=0.3,
        stratify=combined_df["aaChanges"],
        random_state=random_state,
    )
    eval_one, remainder = sklearn.model_selection.train_test_split(
        remainder,
        test_size=2 / 3,
        stratify=remainder["aaChanges"],
        random_state=random_state,
    )
    eval_two, test = sklearn.model_selection.train_test_split(
        remainder,
        test_size=0.5,
        stratify=remainder["aaChanges"],
        random_state=random_state,
    )

    train_split.to_parquet(output_dir / "train.parquet")
    eval_one.to_parquet(output_dir / "eval_one.parquet")
    eval_two.to_parquet(output_dir / "eval_two.parquet")
    test.to_parquet(output_dir / "test.parquet")

    feature_cols = set(combined_df.columns) - {"aaChanges", "virtualBarcode"}
    with open(output_dir / "features.json", "w") as f:
        json.dump(
            {"target_column": "aaChanges", "feature_columns": list(feature_cols)},
            f,
            indent=2,
        )


if __name__ == "__main__":
    fire.Fire(get_splits)
