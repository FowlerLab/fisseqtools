import collections
import json
import os
from typing import Dict

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_count(
    data_csv_path: os.PathLike, key: str = "geno", filter_synonymous: bool = True,
    filter_wildtype: bool = True,
) -> Dict[str, int]:
    data_df = pd.read_csv(data_csv_path)
    
    if filter_synonymous:
        data_df = data_df[data_df["Variant_Class"] != "Synonymous"]
    
    if filter_wildtype:
        data_df = data_df[data_df["Variant_Class"] != "WT"]

    return collections.Counter(list(data_df[key]))


def dump_count(
    data_csv_path: os.PathLike,
    count_json_path: os.PathLike,
    indent: int = 2,
    **kwargs
) -> None:
    with open(count_json_path, "w") as f:
        json.dump(get_count(data_csv_path, **kwargs), f, indent=indent)


def graph_variant_count_dist(
    variant_count: os.PathLike | Dict[str, int],
    title: str = "Variant Frequency Distribution",
    x_label: str = "Variant Cell Count",
    y_label: str = "Variant Cell Count Frequency",
    fig_file_name: str = "variant_counts.png",
) -> None:
    if isinstance(variant_count, str):
        with open(variant_count) as f:
            variant_count = json.load(f)

    count_frequencies = np.bincount(np.array(list(variant_count.values())))
    count_indices = np.arange(len(count_frequencies))

    plt.plot(count_indices[1:], count_frequencies[1:])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fig_file_name, dpi=200)


def get_cum_variant_cell_count(
    variant_count: os.PathLike | Dict[str, int]
) -> np.ndarray:
    """Get the cumulative variant and cell counts for each threshold"""
    if isinstance(variant_count, str):
        with open(variant_count) as f:
            variant_count = json.load(f)

    cell_count_frequencies = np.bincount(np.array(list(variant_count.values())))[::-1]
    cell_counts = np.arange(len(cell_count_frequencies))[::-1]
    total_cells = cell_count_frequencies * cell_counts
    unique = cell_count_frequencies > 0
    cum_cell_variant_count = np.array(
        [
            cell_counts,
            np.cumsum(cell_count_frequencies),
            np.cumsum(total_cells),
        ]
    ).T

    return cum_cell_variant_count, unique


def graph_cum_cell_variant_count(
    variant_count: os.PathLike | Dict[str, int],
    num_sum_rows: int | None = None,
    sum_stride: int = 1,
) -> None:
    cum_cell_variant_count, unique = get_cum_variant_cell_count(variant_count)

    plt.plot(cum_cell_variant_count[:, 0], cum_cell_variant_count[:, 1])
    plt.title("Variant Count vs. Cell Count Threshold")
    plt.xlabel("Cell Count Threshold")
    plt.ylabel("Variant Count")
    plt.savefig("cum_variant_count.png", dpi=200)

    plt.clf()
    plt.plot(cum_cell_variant_count[:, 0], cum_cell_variant_count[:, 2], color="orange")
    plt.title("Cell Count vs. Cell Count Threshold")
    plt.xlabel("Cell Count Threshold")
    plt.ylabel("Cell Count")
    plt.savefig("cum_cell_count.png", dpi=200)

    if num_sum_rows is not None:
        summary = cum_cell_variant_count[unique]
        summary = summary[::sum_stride]
        summary = summary[: min(num_sum_rows, len(summary))]
        print(summary)


if __name__ == "__main__":
    fire.Fire()
