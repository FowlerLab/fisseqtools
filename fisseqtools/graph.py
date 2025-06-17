import pathlib
import json
import multiprocessing
import random
import re
import warnings
from os import PathLike
from multiprocessing import Pool, cpu_count
from typing import Tuple, List

import fire
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import seaborn as sns
import scipy.cluster.hierarchy
import scipy.spatial.distance
import scipy.stats
import scipy.integrate
import sklearn.decomposition
import sklearn.metrics.pairwise
import sklearn.preprocessing
import tqdm
import umap

# Set a global random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# classify each variant into mutation class
def variant_classification(v):
    """Thanks Sriram"""
    if "fs" in v:
        classification = "Frameshift"
    elif v[-1] == "-":
        vs = v.split("|")
        ncodons_aff = len(vs)
        if ncodons_aff > 2:
            classification = "Other"
        else:
            if ncodons_aff == 1:
                classification = "3nt Deletion"
            elif ncodons_aff == 2:
                if int(vs[0][1:-1]) == (int(vs[1][1:-1]) - 1):
                    classification = "3nt Deletion"
                else:
                    classification = "Other"
            else:
                classification = "Other"
    elif ("X" in v) | ("*" in v):
        classification = "Nonsense"
    elif "WT" in v:
        classification = "WT"
    else:
        regex_match = re.match(r"([A-Z])(\d+)([A-Z])", v)
        if regex_match is None:
            classification = "Other"
        elif regex_match.group(1) == regex_match.group(3):
            classification = "Synonymous"
        else:
            classification = "Single Missense"

    return classification


def graph_score_distribution(
    score_file_path: PathLike,
    variant_class: str | None = None,
    img_save_path: PathLike | None = None,
    experiment_name: str | None = None,
    graph_column: str = "test_roc_auc",
) -> None:
    data_df = pd.read_csv(score_file_path)
    data_df.dropna(inplace=True)
    title: str = "ROC AUC Distribution"
    if experiment_name is not None:
        title = f"{title}: {experiment_name}"

    if variant_class is not None:
        variant_classes = data_df["aaChanges"].apply(variant_classification)
        data_df = data_df[variant_classes == variant_class]
        title += f" ({variant_class})"

    scores = data_df[graph_column]
    mean_score = scores.mean()
    median_score = scores.median()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scores, kde=False, alpha=0.5, stat="density", ax=ax)
    sns.kdeplot(scores, fill=True, ax=ax)

    ax.axvline(
        x=median_score,
        color="black",
        linestyle="--",
        label=f"Median = {median_score:.2f}",
    )
    ax.axvline(
        x=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0, 1)

    if img_save_path:
        fig.savefig(img_save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def graph_score_distribution_by_variant(
    score_file_path: PathLike,
    img_save_path: PathLike | None = None,
    experiment_name: str | None = None,
    graph_column: str = "test_roc_auc",
) -> None:
    data_df = pd.read_csv(score_file_path)
    data_df.dropna(inplace=True)
    data_df["Variant Class"] = data_df["aaChanges"].apply(variant_classification)
    mean_score = data_df[graph_column].mean()

    fig, ax = plt.subplots(figsize=(12, 8))
    violin_plot = sns.violinplot(data=data_df, x="Variant Class", y=graph_column, ax=ax)
    ax.axhline(
        y=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )

    categories = [label.get_text() for label in violin_plot.get_xticklabels()]
    category_counts = data_df["Variant Class"].value_counts().reindex(categories)

    ax.set_xticks(range(len(category_counts)))
    ax.set_xticklabels(
        [f"{cat}\n(n={category_counts[cat]})" for cat in category_counts.index]
    )

    title = "ROC AUC Distribution by Variant Class"
    if experiment_name:
        title += f": {experiment_name}"

    ax.set_title(title)
    ax.set_ylabel("ROC AUC")
    ax.legend()

    if img_save_path:
        fig.savefig(img_save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def graph_auc_examples(
    score_file_path: PathLike,
    variant_class: str | None = None,
    img_save_path: PathLike | None = None,
    experiment_name: str | None = None,
    xlim: int = None,
    graph_column: str = "test_roc_auc",
    window_size: int = 10,
) -> None:
    data_df = pd.read_csv(score_file_path)
    data_df = data_df.dropna()
    title = "Num Training Examples vs. ROC AUC"
    if experiment_name is not None:
        title = f"{title}: {experiment_name}"

    if variant_class is not None:
        variant_classes = data_df["aaChanges"].apply(variant_classification)
        data_df = data_df[variant_classes == variant_class]
        title += f" ({variant_class})"

    title += f" (n = {len(data_df)})"
    example_counts = data_df["Example Count"].to_numpy()
    auc_roc = data_df[graph_column].to_numpy()

    if len(example_counts) > 1:
        xy = np.vstack((example_counts, auc_roc))
        z = scipy.stats.gaussian_kde(xy)(xy)
        spearman, p_val = scipy.stats.pearsonr(example_counts, auc_roc)
    else:
        spearman, p_val = float("NaN"), float("NaN")
        z = np.ones_like(auc_roc)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.scatter(
        example_counts,
        auc_roc,
        c=z,
        cmap="viridis",
        label=f"Spearman={spearman:0.4f}, P={p_val:0.4f}",
        s=10,
    )

    # graph rolling average
    example_counts = data_df["Example Count"].to_numpy()
    auc_roc = data_df[graph_column].to_numpy()
    mean_score = auc_roc.mean()
    idx = np.argsort(example_counts)
    example_counts = example_counts[idx]
    auc_roc = auc_roc[idx]
    mean_auc_roc = (
        pd.Series(auc_roc).rolling(window=window_size, center=True).mean().to_numpy()
    )

    ax.plot(
        example_counts,
        mean_auc_roc,
        label=f"Rolling Average (window = {window_size})",
        color="red",
    )

    ax.axhline(
        y=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )

    ax.set_title(title)
    ax.set_xlabel("Num Training Examples")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0, 1)

    if xlim:
        ax.set_xlim(0, xlim)

    ax.legend()

    if img_save_path:
        fig.savefig(img_save_path)
        plt.close(fig)
    else:
        plt.show()


def graph_single_results(
    score_file_path: PathLike,
    img_save_dir: PathLike,
    experiment_name: str | None = None,
    auc_example_xlim: int | None = None,
    auc_example_window: int = 20,
    graph_column: str = "test_roc_auc",
) -> None:
    img_save_dir = pathlib.Path(img_save_dir)
    variant_classes = [
        None,
        "Frameshift",
        "3nt Deletion",
        "Other",
        "Nonsense",
        "WT",
        "Synonymous",
        "Single Missense",
    ]

    data_df = pd.read_csv(score_file_path)
    data_df.dropna(inplace=True)
    present_classes = set(data_df["aaChanges"].map(variant_classification).unique())
    for variant_class in variant_classes:
        if variant_class is not None and variant_class not in present_classes:
            print(f"Skipping {variant_class}")
            continue

        file_stem_suffix = f"-{variant_class}" if variant_class is not None else ""
        graph_score_distribution(
            score_file_path,
            variant_class=variant_class,
            img_save_path=img_save_dir / f"score_distribution{file_stem_suffix}.png",
            experiment_name=experiment_name,
            graph_column=graph_column,
        )

        graph_auc_examples(
            score_file_path,
            variant_class=variant_class,
            img_save_path=img_save_dir / f"auc_v_examples{file_stem_suffix}.png",
            experiment_name=experiment_name,
            xlim=auc_example_xlim,
            graph_column=graph_column,
            window_size=auc_example_window,
        )

    graph_score_distribution_by_variant(
        score_file_path,
        img_save_path=img_save_dir / "score_violin.png",
        experiment_name=experiment_name,
        graph_column=graph_column,
    )


def graph_one_v_other(
    score_file_path_one: PathLike,
    score_file_path_two: PathLike,
    img_save_path: PathLike | None = None,
    variant_class: str | None = None,
    name_one: str | None = None,
    name_two: str | None = None,
) -> None:
    title = "AUC ROC Score Comparison"
    if name_one is not None and name_two is not None:
        title += f": ({name_one} vs. {name_two})"

    score_df_one = pd.read_csv(score_file_path_one)
    score_df_two = pd.read_csv(score_file_path_two)
    data_df = pd.merge(score_df_one, score_df_two, on="aaChanges", how="inner")

    if variant_class is not None:
        variant_classes = data_df["aaChanges"].apply(variant_classification)
        data_df = data_df[variant_classes == variant_class]
        title += f" ({variant_class})"

    roc_auc_x = data_df["eval_roc_auc_x"].to_numpy()
    roc_auc_y = data_df["eval_roc_auc_y"].to_numpy()

    xy = np.vstack((roc_auc_x, roc_auc_y))
    z = scipy.stats.gaussian_kde(xy)(xy)
    spearman, p_val = scipy.stats.pearsonr(roc_auc_x, roc_auc_y)

    plt.scatter(
        roc_auc_x, roc_auc_y, c=z, label=f"Spearman={spearman:0.4f}, P={p_val:0.4f}"
    )
    plt.title(title)
    plt.xlabel(f"ROC AUC: {name_one}")
    plt.ylabel(f"ROC AUC: {name_two}")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if img_save_path is None:
        plt.show()
    else:
        plt.savefig(img_save_path)


def _finalize_pca_plot_stacked(
    fig: plt.Figure,
    ax_class: plt.Axes,
    ax_dist: plt.Axes,
    title: str,
    img_save_path: PathLike | None,
) -> None:
    ax_class.set_title(title)

    for ax in [ax_class, ax_dist]:
        ax.set_xlabel("UMAP One")
        ax.set_ylabel("UMAP Two")

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path, dpi=150)


def _finalize_pca_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    img_save_path: PathLike | None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("UMAP One")
    ax.set_ylabel("UMAP Two")

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path, dpi=150)


def _color_by_class(
    data_df: pd.DataFrame,
    shap_scores: np.ndarray,
    ax: plt.Axes,
) -> None:
    data_df["Variant Class"] = data_df["aaChanges"].apply(variant_classification)

    variant_type_palette = {
        "Single Missense": "grey",
        "Synonymous": "darkgreen",
        "WT": "red",
        "Frameshift": "purple",
        "3nt Deletion": "grey",
        "Nonsense": "purple",
        "Other": "grey",
    }

    sns.scatterplot(
        data=data_df,
        x=shap_scores[:, 0].flatten(),
        y=shap_scores[:, 1].flatten(),
        hue="Variant Class",
        palette=variant_type_palette,
        s=40,
        ax=ax,
    )


def _color_by_distance(
    data_df: pd.DataFrame,
    shap_scores: np.ndarray,
    distance_measure: str,
    fig: plt.Figure,
    ax: plt.Axes,
) -> None:
    sns.scatterplot(
        x=shap_scores[:, 0].flatten(),
        y=shap_scores[:, 1].flatten(),
        hue=data_df[distance_measure],
        palette="Blues",
        ax=ax,
        legend=False,
    )

    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0.5, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(distance_measure)


def _pca(shap_scores: np.ndarray, pca_n_components: int) -> np.ndarray:
    shap_scores = sklearn.preprocessing.StandardScaler().fit_transform(shap_scores)
    pca = sklearn.decomposition.PCA(n_components=pca_n_components, random_state=SEED)
    shap_scores = pca.fit_transform(shap_scores)

    return shap_scores


def _cluster_reduction(shap_scores: np.ndarray, cluster_assignments: int) -> None:
    num_clusters = cluster_assignments.max()
    cluster_matrix = np.eye(num_clusters)[cluster_assignments - 1]
    return shap_scores @ cluster_matrix


def _prepare_shap_scores(
    shap_file_path: pathlib.Path,
    aggregate: bool,
    cluster_assignments: pathlib.Path | None,
    pca_n_components: int,
    reduce_dimensions: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads and processes SHAP scores from a Parquet file.

    Args:
        shap_file_path (PathLike): Path to the SHAP scores file.
        aggregate (bool): Whether to aggregate data by 'aaChanges'.
        cluster_assignments (PathLike | None): Path to cluster assignments.
        pca_n_components (int): Number of PCA components.
        reduce_dimensions (bool): Whether to reduce dimensions.

    Returns:
        tuple: Processed DataFrame and transformed SHAP scores (numpy array).
    """
    data_df = pd.read_parquet(shap_file_path)
    if aggregate:
        data_df = data_df.groupby("aaChanges", as_index=False).mean()

    shap_scores = data_df.drop(["aaChanges", "p_is_var"], axis=1)
    shap_columns = shap_scores.columns.to_frame(index=False, name="feature")
    shap_scores = shap_scores.to_numpy(dtype=float)

    if reduce_dimensions:
        if cluster_assignments is None:
            shap_scores = _pca(shap_scores, pca_n_components)
        else:
            cluster_df = pd.read_csv(cluster_assignments)
            shap_columns = shap_columns.merge(cluster_df, on="feature")
            cluster_idx = shap_columns["cluster_idx"].to_numpy(dtype=int)
            shap_scores = _cluster_reduction(shap_scores, cluster_idx)

    shap_scores = sklearn.preprocessing.StandardScaler().fit_transform(shap_scores)
    return data_df, shap_scores


def umap_shap(
    shap_file_path: pathlib.Path,
    pca_n_components: int = 50,
    umap_n_neighbors: int = 5,
    umap_min_dist: float = 0.1,
    aggregate: bool = True,
    train_results_path: pathlib.Path | None = None,
    color_by_distance: bool = True,
    color_by_class: bool = True,
    img_save_path: pathlib.Path | None = None,
    cluster_assignments: pathlib.Path | None = None,
    reduce_dimensions: bool = True,
    graph_column: str = "test_roc_auc",
    save_components: bool = True,
    experiment_name: str | None = None,
    stack_plots: bool = True,
) -> None:
    """Processes SHAP scores using UMAP and visualizes results."""
    data_df, shap_scores = _prepare_shap_scores(
        shap_file_path,
        aggregate,
        cluster_assignments,
        pca_n_components,
        reduce_dimensions,
    )

    shap_scores = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=2,
        min_dist=umap_min_dist,
        random_state=SEED,
        metric="cosine",
    ).fit_transform(shap_scores)

    if save_components and img_save_path is not None:
        pd.DataFrame(
            {
                "aaChanges": data_df["aaChanges"],
                "umap_one": shap_scores[:, 0].flatten(),
                "umap_two": shap_scores[:, 1].flatten(),
            }
        ).to_csv(pathlib.Path(img_save_path) / "umap_components.csv")
    elif save_components:
        warnings.warn("img_save_path must be set to save UMAP components.")

    if train_results_path is not None:
        train_res_df = pd.read_csv(train_results_path)
        data_df = data_df.merge(train_res_df, on="aaChanges")

    title = "Variant SHAP Score UMAP" + ("" if not aggregate else " (Aggregated)")
    title = f"{title} ({experiment_name})" if experiment_name else title

    fig_size = (16, 9)
    if stack_plots and color_by_distance and color_by_class:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
        fig_class, ax_class = fig, axes[0]
        fig_dist, ax_dist = fig, axes[1]
    else:
        fig_class, ax_class = plt.subplots(figsize=fig_size)
        fig_dist, ax_dist = plt.subplots(figsize=fig_size)

    if color_by_distance:
        distance_measure = "p_is_var" if train_results_path is None else graph_column
        _color_by_distance(data_df, shap_scores, distance_measure, fig_dist, ax_dist)

        if not stack_plots:
            _finalize_pca_plot(
                fig_dist,
                ax_dist,
                title,
                (
                    None
                    if img_save_path is None
                    else pathlib.Path(img_save_path) / "umap_by_distance.png"
                ),
            )

    if color_by_class:
        _color_by_class(data_df, shap_scores, ax_class)

        if not stack_plots:
            _finalize_pca_plot(
                fig_class,
                ax_class,
                title,
                (
                    None
                    if img_save_path is None
                    else pathlib.Path(img_save_path) / "umap_by_class.png"
                ),
            )

    if stack_plots:
        _finalize_pca_plot_stacked(
            fig,
            ax_class,
            ax_dist,
            title,
            (
                None
                if img_save_path is None
                else pathlib.Path(img_save_path) / "umap_stacked.png"
            ),
        )


def get_feature_clusters(
    data_path: PathLike,
    meta_data_path: PathLike,
    threshold: float = 0.9,
    num_clusters: int = 100,
    use_threshold: bool = True,
    plot: bool = False,
    img_save_path: PathLike | None = None,
) -> None:
    data_df = pd.read_parquet(data_path)
    with open(meta_data_path) as f:
        meta_data = json.load(f)

    feature_columns = meta_data["feature_columns"]
    data_df = data_df[feature_columns]
    data_df = data_df.dropna()
    constant_mask = (data_df.nunique() == 1).to_numpy()
    filt_data_df = data_df.loc[:, ~constant_mask]
    feature_matrix = filt_data_df.to_numpy(dtype=float)

    correlation, _ = scipy.stats.spearmanr(feature_matrix, axis=0)
    correlation = np.nan_to_num(correlation, nan=0)
    correlation = np.abs(correlation)
    distance_matrix = 1 - correlation
    distance_matrix = scipy.spatial.distance.squareform(distance_matrix, checks=False)
    linkage_matrix = scipy.cluster.hierarchy.linkage(distance_matrix, method="average")

    cluster_threshold = 1 - threshold if use_threshold else num_clusters
    criterion = "distance" if use_threshold else "maxclust"
    clusters = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, cluster_threshold, criterion=criterion
    )

    const_cluster_start = clusters.max() + 1
    const_cluster_end = const_cluster_start + constant_mask.sum()

    clusters_df = pd.DataFrame({"feature": feature_columns})
    clusters_df["cluster_idx"] = -1
    clusters_df.loc[~constant_mask, "cluster_idx"] = clusters
    clusters_df.loc[constant_mask, "cluster_idx"] = np.arange(
        const_cluster_start, const_cluster_end
    )

    # Sort cluster assignments by cluster size and then cluster idx
    cluster_counts = clusters_df["cluster_idx"].value_counts()
    clusters_df["cluster_count"] = clusters_df["cluster_idx"].map(cluster_counts)
    clusters_df = clusters_df.sort_values(
        by=["cluster_count", "cluster_idx"], ascending=[False, True]
    )
    clusters_df.to_csv("cluster_assignments.csv")

    print(f"Number of clusters: {const_cluster_end - 1}")
    print(f"Top five most frequent clusters:")
    print(cluster_counts.head(5))

    if img_save_path is not None or plot:
        sns.clustermap(
            correlation,
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            cmap="coolwarm",
            annot=False,
            cbar_kws={"label": "Absolute Spearman Correlation"},
            vmin=0,
            vmax=1,
        )

        if img_save_path is None:
            plt.show()
        else:
            plt.savefig(img_save_path)


def split_replicate(train_results_path: PathLike) -> None:
    train_results_dir = pathlib.Path(train_results_path).parent
    train_results_df = pd.read_csv(train_results_path)
    r1_df = train_results_df[train_results_df["replicate"] == 1]
    r2_df = train_results_df[train_results_df["replicate"] == 2]
    r1_df.to_csv(train_results_dir / "train_results_r1.csv")
    r2_df.to_csv(train_results_dir / "train_results_r2.csv")


def combine_results(
    results_one_dir: PathLike,
    results_two_dir: PathLike,
    results_output_dir: PathLike,
) -> None:
    results_one_dir = pathlib.Path(results_one_dir)
    results_two_dir = pathlib.Path(results_two_dir)
    results_output_dir = pathlib.Path(results_output_dir)

    for file in ["train_results.csv", "test_shap.parquet"]:
        if file.endswith(".csv"):
            load_fun = pd.read_csv
            dump_fun = lambda x: x.to_csv(results_output_dir / file)
        else:
            load_fun = pd.read_parquet
            dump_fun = lambda x: x.to_parquet(results_output_dir / file)

        dump_fun(
            pd.concat(
                (load_fun(results_one_dir / file), load_fun(results_two_dir / file))
            )
        )


def cosine_similarity_matrix(
    data_path: PathLike,
    meta_data_path: PathLike,
    output_file: PathLike,
    demean_only: bool = False,
    shuffle_columns: bool = False,
    n_pca_components: int | None = None,
) -> None:
    data_df = pd.read_parquet(data_path)
    with open(meta_data_path) as f:
        meta_data = json.load(f)

    all_columns = meta_data["feature_columns"] + [meta_data["target_column"]]
    data_df = data_df[all_columns]
    data_df = data_df.groupby(meta_data["target_column"], as_index=False).median()
    data_df = data_df[data_df[meta_data["target_column"]] != "WT"]
    feature_matrix = data_df[meta_data["feature_columns"]].to_numpy()

    if shuffle_columns:
        rng = np.random.default_rng(seed=42)
        for col_idx in range(feature_matrix.shape[1]):
            rng.shuffle(feature_matrix[:, col_idx])

    if isinstance(n_pca_components, int):
        feature_matrix = sklearn.decomposition.PCA(
            n_components=n_pca_components
        ).fit_transform(feature_matrix)

    if demean_only:
        means = np.mean(feature_matrix, axis=0, keepdims=True)
        feature_matrix = feature_matrix - means
    else:
        feature_matrix = sklearn.preprocessing.StandardScaler().fit_transform(
            feature_matrix
        )

    cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(feature_matrix)
    pd.DataFrame(
        cosine_similarity,
        index=data_df[meta_data["target_column"]],
        columns=data_df[meta_data["target_column"]],
    ).to_csv(output_file)

    ranks = np.argsort(cosine_similarity, axis=0)
    pd.DataFrame(
        ranks[::-1],
        index=data_df[meta_data["target_column"]],
        columns=data_df[meta_data["target_column"]],
    ).to_csv("rank_" + output_file)


def plot_feature_distribution(
    data_path: PathLike,
    meta_data_path: PathLike,
    standardize: bool = True,
    title: str = "Standardized feature distribution",
    image_save_path: PathLike | None = None,
) -> None:
    data_df = pd.read_parquet(data_path)
    with open(meta_data_path) as f:
        meta_data = json.load(f)

    all_columns = meta_data["feature_columns"] + [meta_data["target_column"]]
    data_df = data_df[all_columns]
    data_df = data_df.groupby(meta_data["target_column"], as_index=False).median()
    data_df = data_df[data_df[meta_data["target_column"]] != "WT"]
    feature_matrix = data_df[meta_data["feature_columns"]].to_numpy()

    if standardize:
        feature_matrix = sklearn.preprocessing.StandardScaler().fit_transform(
            feature_matrix
        )

    feature_matrix = feature_matrix.flatten()
    fig, ax = plt.subplots()
    sns.histplot(x=feature_matrix, discrete=False, kde=True, ax=ax)
    ax.set_title(title)

    if image_save_path is not None:
        fig.savefig(image_save_path)
    else:
        plt.show()


def get_sparcity(
    data_path: PathLike,
    meta_data_path: PathLike,
    axis=1,
) -> np.ndarray:
    data_df = pd.read_parquet(data_path)
    with open(meta_data_path) as f:
        meta_data = json.load(f)

    all_columns = meta_data["feature_columns"] + [meta_data["target_column"]]
    data_df = data_df[all_columns]
    data_df = data_df.groupby(meta_data["target_column"], as_index=False).median()
    data_df = data_df[data_df[meta_data["target_column"]] != "WT"]
    feature_matrix = data_df[meta_data["feature_columns"]].to_numpy()
    feature_matrix = ~np.isclose(feature_matrix, 0.0)
    feature_matrix = feature_matrix.sum(axis=axis)

    return feature_matrix

def plot_example_sparcity(
    feature_data_path: PathLike,
    shap_data_path: PathLike,
    meta_data_path: PathLike,
    image_save_path: PathLike | None = None,
) -> None:
    feature_sparcity = get_sparcity(feature_data_path, meta_data_path)
    shap_sparcity = get_sparcity(shap_data_path, meta_data_path)
    fig, ax = plt.subplots()
    
    sns.histplot(
        x=feature_sparcity,
        discrete=True,
        kde=True,
        label="Features",
        alpha=0.5,
        color="orange",
        ax=ax,
    )

    sns.histplot(
        x=shap_sparcity,
        discrete=True,
        kde=True,
        label="Shap values",
        alpha=0.5,
        color="blue",
        ax=ax,
    )

    ax.legend()
    ax.set_xlabel("Number of non-zero elemenets")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of number of non-zero elements over feature vectors")

    if image_save_path is not None:
        fig.savefig(image_save_path)
    else:
        plt.show()


def plot_feature_sparcity(
    feature_data_path: PathLike,
    shap_data_path: PathLike,
    meta_data_path: PathLike,
    image_save_path: PathLike | None = None,
) -> None:
    feature_sparcity = get_sparcity(feature_data_path, meta_data_path, axis=0)
    shap_sparcity = get_sparcity(shap_data_path, meta_data_path, axis=0)
    fig, ax = plt.subplots()
    
    sns.histplot(
        x=feature_sparcity,
        discrete=True,
        kde=True,
        label="Features",
        alpha=0.5,
        color="orange",
        ax=ax,
    )

    """
    sns.histplot(
        x=shap_sparcity,
        discrete=True,
        kde=True,
        label="Shap values",
        alpha=0.5,
        color="blue",
        ax=ax,
    )
    """

    ax.legend()
    ax.set_xlabel("Number of non-zero elemenets")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of number of non-zero elements over feature columns")

    if image_save_path is not None:
        fig.savefig(image_save_path)
    else:
        plt.show()


def process_variant(
    args: Tuple[str, np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[str, List[np.ndarray]]:
    curr_var, aa_changes, data_col, shap_col = args
    data_row = aa_changes[data_col]
    shap_row = aa_changes[shap_col]
    shared_neighbors = [
        len(np.intersect1d(data_row[: i + 1], shap_row[: i + 1]))
        for i in range(len(aa_changes))
    ]

    return curr_var, shared_neighbors


def rank_changes(data_rank_path: PathLike, shap_rank_path: PathLike) -> None:
    data_df = pd.read_csv(data_rank_path)
    shap_df = pd.read_csv(shap_rank_path)

    aa_changes = data_df["aaChanges"].to_numpy()
    data_df = data_df.drop("aaChanges", axis=1)
    shap_df = shap_df.drop("aaChanges", axis=1)

    rank_changes_arr = np.abs(data_df.to_numpy() - shap_df.to_numpy())
    pd.DataFrame(
        {
            "aaChanges": aa_changes,
            "median_rank_change": np.median(rank_changes_arr, axis=0).flatten(),
            "mean_rank_change": np.mean(rank_changes_arr, axis=0).flatten(),
        }
    ).to_csv("rank_change.csv", index=False)

    variant_names = data_df.columns.tolist()
    _, aa_idx = np.unique(aa_changes, return_inverse=True)
    args_list = [
        (
            var,
            aa_idx,
            data_df[var].to_numpy().astype(int),
            shap_df[var].to_numpy().astype(int),
        )
        for var in variant_names
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(process_variant, args_list),
                total=len(args_list),
                desc="processing variants",
            )
        )

    rank_change_dict = dict(results)
    results_dict = {"num_neighbors": list(range(1, len(aa_changes) + 1))}
    results_dict.update(rank_change_dict)
    pd.DataFrame(results_dict).to_csv("shared_neighbors.csv", index=False)


def var_v_neighborhood_size(
    rank_matrix_path: PathLike, img_save_path: PathLike = None
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy()
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)
    variances = np.std(data_matrix, axis=1)
    means = np.mean(data_matrix, axis=1)
    adjusted_dist = variances / means

    fig, ax = plt.subplots()

    ax.plot(num_neighbors, adjusted_dist)
    ax.set_ylim(bottom=0)

    max_idx = np.argmax(adjusted_dist)
    ax.axvline(
        num_neighbors[max_idx], color="red", linestyle="-", label=f"max idx = {max_idx}"
    )
    ax.legend()

    ax.set_ylabel("STD / Mean")
    ax.set_xlabel("Neighborhood Size")
    ax.set_title(
        "Neighborhood Size vs. Adjusted Standard Deviation of Shared Neighbors"
    )

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def plot_mean_proportion_shared_neighbors(
    rank_matrix_path: PathLike, img_save_path: PathLike = None
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy()
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)
    data_matrix = data_matrix / num_neighbors[:, np.newaxis]
    adjusted_variances = np.std(data_matrix, axis=1)
    adjusted_means = np.mean(data_matrix, axis=1)
    max_var_idx = num_neighbors[np.argmax(adjusted_variances)]

    fig, ax = plt.subplots()

    ax.plot(num_neighbors, adjusted_means, color="blue", label="mean shared proportion")
    ax.plot(
        num_neighbors,
        adjusted_means + adjusted_variances,
        color="black",
        linestyle="--",
        label="+1 std",
    )
    ax.plot(
        num_neighbors,
        adjusted_means - adjusted_variances,
        color="black",
        linestyle="--",
        label="-1 std",
    )
    ax.axvline(max_var_idx, color="red", label=f"max std idx = {max_var_idx}")

    ax.set_title("Mean Proportion of Neighbors that are Shared")
    ax.set_xlabel("Neighborhood Size")
    ax.set_ylabel("Proportion Shared Neighbors")
    ax.legend()

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def plot_mean_shared_neighbors(
    rank_matrix_path: PathLike, img_save_path: PathLike = None
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy()
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)
    variances = np.std(data_matrix, axis=1)
    means = np.mean(data_matrix, axis=1)

    fig, ax = plt.subplots()

    ax.plot(num_neighbors, means, color="blue", label="mean shared neighbors")
    ax.plot(
        num_neighbors,
        means + variances,
        color="black",
        linestyle="--",
        label="+1 std",
    )
    ax.plot(
        num_neighbors,
        means - variances,
        color="black",
        linestyle="--",
        label="-1 std",
    )
    ax.plot(
        np.linspace(0, num_neighbors.max(), 100),
        np.linspace(0, num_neighbors.max(), 100),
        color="grey",
        linestyle="--",
        label="y = x",
    )

    ax.set_title("Mean Shared Neigbors")
    ax.set_xlabel("Neighborhood Size")
    ax.set_ylabel("Number Shared Neighbors")
    ax.legend()

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def plot_median_proportion_shared_neighbors(
    rank_matrix_path: PathLike, img_save_path: PathLike = None
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy()
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)
    data_matrix = data_matrix / num_neighbors[:, np.newaxis]
    adjusted_upper_quantile = np.quantile(data_matrix, 0.75, axis=1)
    adjusted_lower_quantile = np.quantile(data_matrix, 0.25, axis=1)
    diff = adjusted_upper_quantile - adjusted_lower_quantile
    adjusted_median = np.median(data_matrix, axis=1)
    max_var_idx = num_neighbors[np.argmax(diff)]

    fig, ax = plt.subplots()

    ax.plot(
        num_neighbors, adjusted_median, color="blue", label="median shared proportion"
    )
    ax.plot(
        num_neighbors,
        adjusted_upper_quantile,
        color="black",
        linestyle="--",
        label="upper quantile",
    )
    ax.plot(
        num_neighbors,
        adjusted_lower_quantile,
        color="black",
        linestyle="--",
        label="lower quantile",
    )
    ax.axvline(
        max_var_idx,
        color="red",
        label=f"max quantile diff idx = {max_var_idx}",
    )

    ax.set_title("Median Proportion of Neighbors that are Shared")
    ax.set_xlabel("Neighborhood Size")
    ax.set_ylabel("Proportion Shared Neighbors")
    ax.legend()

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def plot_median_shared_neighbors(
    rank_matrix_path: PathLike, img_save_path: PathLike = None
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy()
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)
    upper_quantile = np.quantile(data_matrix, 0.75, axis=1)
    lower_quantile = np.quantile(data_matrix, 0.25, axis=1)
    adjusted_median = np.median(data_matrix, axis=1)

    fig, ax = plt.subplots()

    ax.plot(
        num_neighbors, adjusted_median, color="blue", label="median shared neighbors"
    )
    ax.plot(
        num_neighbors,
        upper_quantile,
        color="black",
        linestyle="--",
        label="upper quantile",
    )
    ax.plot(
        num_neighbors,
        lower_quantile,
        color="black",
        linestyle="--",
        label="lower quantile",
    )
    ax.plot(
        np.linspace(0, num_neighbors.max(), 100),
        np.linspace(0, num_neighbors.max(), 100),
        color="grey",
        linestyle="--",
        label="y = x",
    )

    ax.set_title("Median Shared Neighbors")
    ax.set_xlabel("Neighborhood Size")
    ax.set_ylabel("Nuber of Shared Neighbors")
    ax.legend()

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def graph_num_shared_neighbors(
    rank_matrix_path: PathLike, row: int, img_save_path: PathLike = None
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy()
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)

    fig = plt.figure(figsize=(8, 6))
    ax_hist = fig.add_axes([0.1, 0.3, 0.8, 0.65])
    ax_slider = fig.add_axes([0.1, 0.1, 0.8, 0.05])

    n_size_slider = matplotlib.widgets.Slider(
        ax=ax_slider,
        label="Neighborhood Size",
        valinit=row,
        valmin=num_neighbors.min(),
        valmax=num_neighbors.max(),
    )

    def update(val: float) -> None:
        ax_hist.clear()
        data_row = data_matrix[int(val)]
        sns.histplot(x=data_row, discrete=True, ax=ax_hist, kde=True)
        ax_hist.set_xlabel("Num Shared Neighbors")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title(f"Distribution of Shared Neighbors (n = {int(val)})")
        fig.canvas.draw_idle()

    update(row)
    n_size_slider.on_changed(update)

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def int_shared_neighbor(
    rank_matrix_path: PathLike,
    img_save_path: PathLike = None,
    auc_save_path: PathLike = None,
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy().reshape((-1, 1))
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=int)
    int_neighbors = np.trapz(y=data_matrix, x=num_neighbors, axis=0)
    int_neighbors = (2 * int_neighbors) / (data_matrix.shape[1] ** 2)

    fig, ax = plt.subplots()
    sns.histplot(x=int_neighbors, discrete=False, ax=ax)

    if auc_save_path is not None:
        pd.DataFrame(int_neighbors, index=data_df.columns).to_csv(auc_save_path)

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def plot_aggregated_proportion(
    rank_matrix_path: PathLike,
    img_save_path: PathLike = None,
) -> None:
    data_df = pd.read_csv(rank_matrix_path)
    num_neighbors = data_df["num_neighbors"].to_numpy(dtype=float)
    data_df.drop("num_neighbors", axis=1, inplace=True)
    data_matrix = data_df.to_numpy(dtype=float)
    num_neighbors = num_neighbors.reshape((-1, 1))
    data_matrix /= num_neighbors
    data_matrix = data_matrix.mean(axis=0)

    fig, ax = plt.subplots()
    sns.histplot(x=data_matrix, discrete=False, ax=ax)
    ax.set_title("Distribution of Mean Shared Proportion")

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def run_all_neighbor_plots(
    rank_matrix_path: PathLike,
    output_dir: PathLike = None,
) -> None:
    """Run all neighbor analysis plots."""
    rank_matrix_path = pathlib.Path(rank_matrix_path)

    if output_dir is None:
        save_paths = [None] * 6
    else:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_paths = [
            output_dir / "var_v_neighborhood_size.png",
            output_dir / "mean_proportion_shared_neighbors.png",
            output_dir / "mean_shared_neighbors.png",
            output_dir / "median_proportion_shared_neighbors.png",
            output_dir / "median_shared_neighbors.png",
            output_dir / "agregated_proportion.png",
        ]

    var_v_neighborhood_size(rank_matrix_path, save_paths[0])
    plot_mean_proportion_shared_neighbors(rank_matrix_path, save_paths[1])
    plot_mean_shared_neighbors(rank_matrix_path, save_paths[2])
    plot_median_proportion_shared_neighbors(rank_matrix_path, save_paths[3])
    plot_median_shared_neighbors(rank_matrix_path, save_paths[4])
    plot_aggregated_proportion(rank_matrix_path, save_paths[5])


def aggregated_feature_clusters(
    data_path: PathLike,
    meta_data_path: PathLike,
) -> None:
    data_df = pd.read_parquet(data_path)
    with open(meta_data_path) as f:
        meta_data = json.load(f)

    all_columns = meta_data["feature_columns"] + [meta_data["target_column"]]
    data_df = data_df[all_columns]
    data_df = data_df.groupby(meta_data["target_column"], as_index=False).median()
    data_df = data_df[data_df[meta_data["target_column"]] != "WT"]
    data_df = data_df[meta_data["feature_columns"]]
    data_df = data_df.loc[:, data_df.nunique() > 1]
    correlation_matrix = data_df.corr(method="spearman").to_numpy()
    correlation_matrix = np.abs(correlation_matrix)

    cluster_map = sns.clustermap(correlation_matrix, vmin=0, vmax=1)
    cluster_map.figure.suptitle("Absolute Shap Correlation")
    plt.show()

     
def main():
    fire.Fire()


if __name__ == "__main__":
    main()
