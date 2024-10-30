import collections
import json
import math
import os
import pathlib
import pickle
from typing import Dict

import fire
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import scipy.sparse
import scipy.spatial.distance
import scipy.stats
import sklearn.feature_selection
import sklearn.preprocessing
import sklearn.model_selection
import tqdm


def get_count(
    data_csv_path: os.PathLike,
    key: str = "geno",
    filter_synonymous: bool = True,
    filter_wildtype: bool = True,
) -> Dict[str, int]:
    data_df = pd.read_csv(data_csv_path)

    if filter_synonymous:
        data_df = data_df[data_df["Variant_Class"] != "Synonymous"]

    if filter_wildtype:
        data_df = data_df[data_df["Variant_Class"] != "WT"]

    return collections.Counter(list(data_df[key]))


def dump_count(
    data_csv_path: os.PathLike, count_json_path: os.PathLike, indent: int = 2, **kwargs
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


def graph_feature_correlation(
    feature_matrix_pkl: os.PathLike,
    fig_file_name: os.PathLike,
    sample: float | None = None,
) -> None:
    print("Loading Features...")
    with open(feature_matrix_pkl, "rb") as f:
        feature_stack: np.ndarray = pickle.load(f)

    feature_stack = feature_stack[:, 384:769]
    if sample:
        num_rows = feature_stack.shape[0]
        feature_stack = feature_stack[
            np.random.choice(num_rows, math.ceil(num_rows * sample), replace=False), :
        ]

    print("Calculating Spearman Correlation...")
    spearman_corr = np.corrcoef(feature_stack.T)
    abs_spearman_corr = np.abs(spearman_corr)

    # Reorder clusters via hierarchal clustering
    print("Clustering...")
    linkage_mat = scipy.cluster.hierarchy.linkage(abs_spearman_corr, method="average")
    dendrogram = scipy.cluster.hierarchy.dendrogram(linkage_mat, no_plot=True)
    cluster_indices = dendrogram["leaves"]
    spearman_corr = spearman_corr[cluster_indices, :][:, cluster_indices]

    print("Saving Image...")
    cmap = cm.get_cmap("coolwarm")
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    plt.imshow(spearman_corr, cmap=cmap, norm=norm, aspect="equal")
    plt.title("Feature Correlation With Hierarchical Clustering Reordering")
    plt.colorbar()
    plt.savefig(fig_file_name)


def get_mutual_info(
    data_path: os.PathLike,
    feature_matrix_pkl: os.PathLike,
    save_dir: os.PathLike,
    sample: float | None = None,
) -> None:
    save_path = pathlib.Path(save_dir)
    data_df = pd.read_csv(data_path)
    with open(feature_matrix_pkl, "rb") as f:
        features: np.ndarray = pickle.load(f)

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(data_df["geno"])
    labels = label_encoder.transform(data_df["geno"])
    select_features = features[data_df["embedding_index"].to_numpy()]

    if sample:
        sss = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=1 - sample
        )
        sample_idx = next(sss.split(select_features, labels))
        labels = labels[sample_idx]
        select_features = select_features[sample_idx]

    mutual_information = sklearn.feature_selection.mutual_info_classif(
        select_features, labels
    )
    np.save(save_path / "me-all.npy", mutual_information)

    unique_labels = np.sort(np.unique(labels))
    me_matrix = np.zeros((len(labels), features.shape[0]), dtype=float)

    for curr_label in tqdm.tqdm(unique_labels, desc="Computing ME Matrix"):
        curr_label_vec = np.zeros(len(labels), dtype=int)
        curr_label_vec[curr_label_vec == curr_label] = 1
        me_matrix[curr_label] = sklearn.feature_selection.mutual_info_classif(
            select_features, curr_label_vec
        )

    np.save(save_path / "me-varmat.npy", me_matrix)
    np.save(
        save_path / "label-vec.npy",
        np.array(label_encoder.inverse_transform(unique_labels)),
    )


if __name__ == "__main__":
    fire.Fire()
