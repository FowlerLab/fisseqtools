import pandas as pd

from fisseqtools.simple_split import filter_single_replicate, get_splits


def test_filter_single_replicate():
    data_df = pd.DataFrame(
        {
            "aaChanges": ["A"] * 3 + ["B"] * 2 + ["C"] * 1 + ["D"] * 4,
            "value": list(range(10)),
            "barcode": ["AA"] * 5 + ["AB"] * 5,
        }
    )

    filtered_df = filter_single_replicate(data_df, 3)
    filtered_df_counts = filtered_df["aaChanges"].value_counts()

    assert len(filtered_df) == 7
    assert "A" in filtered_df_counts
    assert "D" in filtered_df_counts
    assert "B" not in filtered_df_counts
    assert "C" not in filtered_df_counts
    assert filtered_df_counts["A"] == 3
    assert filtered_df_counts["D"] == 4


def test_get_splits(tmp_path):
    data_df = pd.DataFrame(
        {"aaChanges": ["A"] * 10 + ["B"] * 10 + ["C"] * 2, "value": list(range(22))}
    )
    data_df_path = tmp_path / "data.parquet"
    data_df.to_parquet(data_df_path)

    output_dir = tmp_path / "no_filter"
    output_dir.mkdir()
    get_splits(data_df_path, output_dir, min_count=10)

    assert (output_dir / "train.parquet").is_file()
    assert (output_dir / "eval.parquet").is_file()
    assert (output_dir / "test.parquet").is_file()

    train = pd.read_parquet(output_dir / "train.parquet")
    eval = pd.read_parquet(output_dir / "eval.parquet")
    test = pd.read_parquet(output_dir / "test.parquet")

    assert len(train) == 16
    assert len(eval) == 2
    assert len(test) == 2

    assert train["aaChanges"].value_counts().to_dict() == {"A": 8, "B": 8}
    assert eval["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}
    assert test["aaChanges"].value_counts().to_dict() == {"A": 1, "B": 1}
