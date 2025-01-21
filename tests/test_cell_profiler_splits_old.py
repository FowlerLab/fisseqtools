import pandas as pd

from fisseqtools.archive.cell_profiler_splits_old import (
    filter_no_stratify,
    filter_non_numeric,
    filter_stratify_replicates,
    get_splits,
)


def test_filter_non_numeric():
    r1_df = pd.DataFrame(
        {
            "aaChanges": ["A"] * 3 + ["B"] * 2 + ["C"] * 1 + ["D"] * 4,
            "value": list(range(10)),
            "barcode": ["AA"] * 5 + ["AB"] * 5,
        }
    )

    r1_df_filt = filter_non_numeric(r1_df).reset_index(drop=True)
    assert r1_df.shape[1] == 3  # Include index column
    assert "barcode" not in r1_df_filt.columns
    assert "value" in r1_df_filt.columns
    assert "aaChanges" in r1_df_filt.columns


def test_filter_funs():
    r1_df = pd.DataFrame(
        {
            "aaChanges": ["A"] * 3 + ["B"] * 2 + ["C"] * 1 + ["D"] * 4,
            "value": list(range(10)),
            "barcode": ["AA"] * 5 + ["AB"] * 5,
        }
    )
    r2_df = pd.DataFrame(
        {
            "aaChanges": ["A"] * 2 + ["B"] * 3 + ["C"] * 1 + ["E"] * 1,
            "value": list(range(10, 17)),
            "barcode": ["AA"] * 5 + ["AB"] * 2,
        }
    )

    result = filter_stratify_replicates(r1_df, r2_df, min_count=2)

    assert "barcode" not in result.columns
    assert "value" in result.columns
    assert "aaChanges" in result.columns

    counts = result["aaChanges"].value_counts()
    assert counts["A"] == 4
    assert counts["B"] == 4
    assert counts["C"] == 2
    assert "D" not in counts.index
    assert "E" not in counts.index

    result = filter_no_stratify(r1_df, r2_df, min_count=2)

    assert "barcode" not in result.columns
    assert "value" in result.columns
    assert "aaChanges" in result.columns

    counts = result["aaChanges"].value_counts()
    assert counts["A"] == 5
    assert counts["B"] == 5
    assert counts["C"] == 2
    assert "D" not in counts.index
    assert "E" not in counts.index


def test_get_splits(tmp_path):
    r1_data = pd.DataFrame(
        {"aaChanges": ["A"] * 5 + ["B"] * 5 + ["C"] * 1, "value": list(range(11))}
    )
    r2_data = pd.DataFrame(
        {"aaChanges": ["A"] * 5 + ["B"] * 5 + ["C"] * 1, "value": list(range(11, 22))}
    )

    # Save to Parquet files
    r1_path = tmp_path / "r1.parquet"
    r2_path = tmp_path / "r2.parquet"
    r1_data.to_parquet(r1_path)
    r2_data.to_parquet(r2_path)
    splits_path = tmp_path / "splits"
    splits_path.mkdir()

    get_splits(
        r1_parquet_path=r1_path,
        r2_parquet_path=r2_path,
        output_dir=splits_path,
        stratify_replicates=False,
        min_count=3,
    )

    assert (splits_path / "train.parquet").exists()
    assert (splits_path / "eval_one.parquet").exists()
    assert (splits_path / "eval_two.parquet").exists()
    assert (splits_path / "test.parquet").exists()

    train = pd.read_parquet(splits_path / "train.parquet")
    eval_one = pd.read_parquet(splits_path / "eval_one.parquet")
    eval_two = pd.read_parquet(splits_path / "eval_two.parquet")
    test = pd.read_parquet(splits_path / "test.parquet")

    assert len(train) == 14
    assert len(eval_one) == 2
    assert len(eval_two) == 2
    assert len(test) == 2

    assert train["aaChanges"].value_counts().to_dict() == {"A": 7, "B": 7}
    assert eval_one["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}
    assert eval_two["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}
    assert test["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}

    splits_path = tmp_path / "splits_two"
    splits_path.mkdir()

    get_splits(
        r1_parquet_path=r1_path,
        r2_parquet_path=r2_path,
        output_dir=splits_path,
        stratify_replicates=True,
        min_count=3,
    )

    assert (splits_path / "train.parquet").exists()
    assert (splits_path / "eval_one.parquet").exists()
    assert (splits_path / "eval_two.parquet").exists()
    assert (splits_path / "test.parquet").exists()

    train = pd.read_parquet(splits_path / "train.parquet")
    eval_one = pd.read_parquet(splits_path / "eval_one.parquet")
    eval_two = pd.read_parquet(splits_path / "eval_two.parquet")
    test = pd.read_parquet(splits_path / "test.parquet")

    assert len(train) == 14
    assert len(eval_one) == 2
    assert len(eval_two) == 2
    assert len(test) == 2

    assert train["aaChanges"].value_counts().to_dict() == {"A": 7, "B": 7}
    assert eval_one["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}
    assert eval_two["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}
    assert test["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}