import pathlib
from os import PathLike

import fire
import numpy as np
import pandas as pd
import sklearn.model_selection

random_state = np.random.RandomState(seed=42)


def filter_single_replicate(data_df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    data_df = data_df.reset_index(drop=True)
    aa_counts = data_df["aaChanges"].value_counts()
    aa_counts = aa_counts[aa_counts >= min_count]
    data_df = data_df[data_df["aaChanges"].isin(aa_counts.index)]

    return data_df


def get_splits(
    data_parquet_path: PathLike,
    output_dir: PathLike,
    min_count: int = 10,
) -> None:
    output_dir = pathlib.Path(output_dir)
    data_df = pd.read_parquet(data_parquet_path)
    filtered_df = filter_single_replicate(data_df, min_count)

    train, remainder = sklearn.model_selection.train_test_split(
        filtered_df,
        test_size=0.2,
        stratify=filtered_df["aaChanges"],
        random_state=random_state,
    )
    eval, test = sklearn.model_selection.train_test_split(
        remainder,
        test_size=0.5,
        stratify=remainder["aaChanges"],
        random_state=random_state,
    )

    train.to_parquet(output_dir / "train.parquet")
    eval.to_parquet(output_dir / "eval.parquet")
    test.to_parquet(output_dir / "test.parquet")


def main() -> None:
    fire.Fire(get_splits)


if __name__ == "__main__":
    main()
