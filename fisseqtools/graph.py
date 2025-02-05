import re
from os import PathLike

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import sklearn.decomposition
import sklearn.preprocessing
import umap


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
) -> None:
    data_df = pd.read_csv(score_file_path)
    title = "Eval ROC AUC Distribution: XGboost w/ Cell Profiler Features"

    if variant_class is not None:
        variant_classes = data_df["aaChanges"].apply(variant_classification)
        data_df = data_df[variant_classes == variant_class]
        title += f" ({variant_class})"

    scores = data_df["eval_roc_auc"]
    mean_score = scores.mean()

    sns.histplot(scores, kde=False, alpha=0.5, stat="density")
    sns.kdeplot(scores, shade=True)
    plt.axvline(
        x=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )
    plt.legend()
    plt.title(title)
    plt.xlabel("ROC AUC")

    if img_save_path is None:
        plt.show()
    else:
        plt.savefig(img_save_path)


def graph_score_distribution_by_variant(
    score_file_path: PathLike, img_save_path: PathLike | None = None
) -> None:
    data_df = pd.read_csv(score_file_path)
    data_df["Variant Class"] = data_df["aaChanges"].apply(variant_classification)

    mean_score = data_df["eval_roc_auc"].mean()
    violin_plot = sns.violinplot(data=data_df, x="Variant Class", y="eval_roc_auc")
    plt.axhline(
        y=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )

    categories = violin_plot.get_xticklabels()
    category_labels = [label.get_text() for label in categories]
    category_counts = data_df["Variant Class"].value_counts().reindex(category_labels)
    plt.xticks(
        ticks=range(len(category_counts)),
        labels=[f"{cat}\n(n={category_counts[cat]})" for cat in category_counts.index],
    )

    plt.legend()
    plt.title("Eval ROC AUC Distribution by Variant Class")
    plt.ylabel("ROC AUC")

    if img_save_path is None:
        plt.show()
    else:
        plt.savefig(img_save_path)


def graph_auc_examples(
    score_file_path: PathLike,
    variant_class: str | None = None,
    img_save_path: PathLike | None = None,
) -> None:
    data_df = pd.read_csv(score_file_path)
    title = "Num Training Examples vs. ROC AUC"

    if variant_class is not None:
        variant_classes = data_df["aaChanges"].apply(variant_classification)
        data_df = data_df[variant_classes == variant_class]
        title += f" ({variant_class})"

    title += f" (n = {len(data_df)})"
    example_counts = data_df["Example Count"].to_numpy()
    auc_roc = data_df["eval_roc_auc"].to_numpy()

    xy = np.vstack((example_counts, auc_roc))
    z = scipy.stats.gaussian_kde(xy)(xy)
    spearman, p_val = scipy.stats.pearsonr(example_counts, auc_roc)

    plt.scatter(
        example_counts, auc_roc, c=z, label=f"Spearman={spearman:0.4f}, P={p_val:0.4f}"
    )
    plt.title(title)
    plt.xlabel("Num Training Examples")
    plt.ylabel("ROC AUC")
    plt.legend()

    if img_save_path is None:
        plt.show()
    else:
        plt.savefig(img_save_path)


def _color_by_class(data_df: pd.DataFrame, shap_scores: np.ndarray) -> None:
    data_df["Variant Class"] = data_df["aaChanges"].apply(variant_classification)
    for curr_class in data_df["Variant Class"].unique():
        mask = (data_df["Variant Class"] == curr_class).to_numpy(dtype=bool)
        target_umap = shap_scores[mask]
        plt.scatter(
            target_umap[:, 0].flatten(),
            target_umap[:, 1].flatten(),
            edgecolors="black",
            linewidth=0.5,
            label=curr_class,
        )

    plt.legend()


def _color_by_distance(
    data_df: pd.DataFrame, shap_scores: np.ndarray, distance_measure: np.ndarray
) -> None:
    plt.scatter(
        shap_scores[:, 0].flatten(),
        shap_scores[:, 1].flatten(),
        c=data_df[distance_measure],
        vmax=1.0,
        vmin=0.5,
        cmap=plt.cm.Blues,
    )
    plt.colorbar()


def _finalize_pca_plot(title: str, img_save_path: PathLike | None) -> None:
    plt.title(title)
    plt.xlabel("UMAP One")
    plt.ylabel("UMAP Two")

    if img_save_path is None:
        plt.show()
    else:
        plt.savefig(img_save_path)


def pca_shap(
    shap_file_path: PathLike,
    pca_n_components: int = 100,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    aggregate: bool = True,
    train_results_path: PathLike | None = None,
    color_by_distance: bool = True,
    color_by_class: bool = True,
    img_save_path: PathLike | None = None,
) -> None:
    data_df = pd.read_parquet(shap_file_path)
    if aggregate:
        data_df = data_df.groupby("aaChanges", as_index=False).mean()

    shap_scores = data_df.drop(["aaChanges", "p_is_var"], axis=1).to_numpy(dtype=float)
    shap_scores = sklearn.preprocessing.StandardScaler().fit_transform(shap_scores)
    shap_scores = sklearn.decomposition.PCA(
        n_components=pca_n_components
    ).fit_transform(shap_scores)
    shap_scores = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=2,
        min_dist=umap_min_dist,
    ).fit_transform(shap_scores)

    if train_results_path is not None:
        train_res_df = pd.read_csv(train_results_path)
        data_df = data_df.merge(train_res_df, on="aaChanges")

    title = "Variant SHAP Score UMAP" + ("" if not aggregate else " (Aggregated)")
    if color_by_distance:
        distance_measure = "p_is_var" if train_results_path is None else "eval_roc_auc"
        _color_by_distance(data_df, shap_scores, distance_measure)
        _finalize_pca_plot(title, img_save_path)

    if color_by_class:
        _color_by_class(data_df, shap_scores)
        _finalize_pca_plot(title, img_save_path)


if __name__ == "__main__":
    fire.Fire()
