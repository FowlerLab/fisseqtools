import pathlib
import json
import random
import re
from os import PathLike

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import scipy.sparse
import seaborn as sns
import scipy.cluster.hierarchy
import scipy.spatial.distance
import scipy.stats
import sklearn.decomposition
import sklearn.preprocessing
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
    sns.kdeplot(scores, shade=True, ax=ax)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        example_counts,
        auc_roc,
        c=z,
        cmap="viridis",
        label=f"Spearman={spearman:0.4f}, P={p_val:0.4f}",
    )

    ax.set_title(title)
    ax.set_xlabel("Num Training Examples")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0, 1)

    if xlim:
        ax.set_xlim(0, xlim)

    ax.legend()

    if img_save_path:
        fig.savefig(img_save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def graph_single_results(
    score_file_path: PathLike,
    img_save_dir: PathLike,
    experiment_name: str | None = None,
    auc_example_xlim: int = None,
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


def _finalize_pca_plot(
    fig: plt.Figure, ax: plt.Axes, title: str, img_save_path: PathLike | None
) -> None:
    ax.set_title(title)
    ax.set_xlabel("UMAP One")
    ax.set_ylabel("UMAP Two")

    if img_save_path is None:
        plt.show()
        plt.close(fig)
    else:
        fig.savefig(img_save_path)


def _color_by_class(
    data_df: pd.DataFrame,
    shap_scores: np.ndarray,
    title: str,
    img_save_path: PathLike | None,
) -> None:
    data_df["Variant Class"] = data_df["aaChanges"].apply(variant_classification)

    variant_type_palette = {
        "Single Missense": "grey",
        "Synonymous": "darkgreen",
        "WT": "grey",
        "Frameshift": "purple",
        "3nt Deletion": "grey",
        "Nonsense": "purple",
        "Other": "grey",
    }

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=data_df,
        x=shap_scores[:, 0].flatten(),
        y=shap_scores[:, 1].flatten(),
        hue="Variant Class",
        palette=variant_type_palette,
        s=40,
    )

    _finalize_pca_plot(fig, ax, title, img_save_path)


def _color_by_distance(
    data_df: pd.DataFrame,
    shap_scores: np.ndarray,
    distance_measure: str,
    title: str,
    img_save_path: PathLike | None,
) -> None:
    fig, ax = plt.subplots()

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

    _finalize_pca_plot(fig, ax, title, img_save_path)


def _pca(shap_scores: np.ndarray, pca_n_components: int) -> np.ndarray:
    shap_scores = sklearn.preprocessing.StandardScaler().fit_transform(shap_scores)
    pca = sklearn.decomposition.PCA(n_components=pca_n_components, random_state=SEED)
    shap_scores = pca.fit_transform(shap_scores)

    return shap_scores


def _cluster_reduction(shap_scores: np.ndarray, cluster_assignments: int) -> None:
    num_clusters = cluster_assignments.max()
    cluster_matrix = np.eye(num_clusters)[cluster_assignments - 1]
    return shap_scores @ cluster_matrix


def umap_shap(
    shap_file_path: PathLike,
    pca_n_components: int = 50,
    umap_n_neighbors: int = 5,
    umap_min_dist: float = 0.1,
    aggregate: bool = True,
    train_results_path: PathLike | None = None,
    color_by_distance: bool = True,
    color_by_class: bool = True,
    img_save_path: PathLike | None = None,
    cluster_assignments: PathLike | None = None,
    reduce_dimensions: bool = True,
    graph_column: str = "test_roc_auc",
    components_save_path: str | None = None,
    experiment_name: str | None = None,
) -> None:
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
    shap_scores = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=2,
        min_dist=umap_min_dist,
        random_state=SEED,  # Set UMAP random seed for reproducibility
        metric="cosine",
    ).fit_transform(shap_scores)

    if components_save_path is not None:
        pd.DataFrame(
            {
                "aaChanges": data_df["aaChanges"],
                "umap_one": shap_scores[:, 0].flatten(),
                "umap_two": shap_scores[:, 1].flatten(),
            }
        ).to_csv(components_save_path)

    if train_results_path is not None:
        train_res_df = pd.read_csv(train_results_path)
        data_df = data_df.merge(train_res_df, on="aaChanges")

    title = "Variant SHAP Score UMAP" + ("" if not aggregate else " (Aggregated)")
    title = f"{title} ({experiment_name})" if experiment_name else title

    if color_by_distance:
        curr_save_path = None
        if img_save_path is not None:
            curr_save_path = pathlib.Path(img_save_path) / "umap_by_distance.png"

        distance_measure = "p_is_var" if train_results_path is None else graph_column
        _color_by_distance(
            data_df, shap_scores, distance_measure, title, curr_save_path
        )

    if color_by_class:
        curr_save_path = None
        if img_save_path is not None:
            curr_save_path = pathlib.Path(img_save_path) / "umap_by_class.png"

        _color_by_class(data_df, shap_scores, title, curr_save_path)


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


def main():
    fire.Fire()


if __name__ == "__main__":
    main()
