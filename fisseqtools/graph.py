import re
from os import PathLike

import fire
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    score_file_path: PathLike, img_save_path: PathLike | None = None
) -> None:
    data_df = pd.read_csv(score_file_path)
    scores = data_df["eval_two_roc_auc"]
    mean_score = scores.mean()

    sns.kdeplot(scores, shade=True)
    plt.axvline(
        x=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )
    plt.legend()
    plt.title("Eval ROC AUC Distribution: XGboost w/ Cell Profiler Features")
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

    mean_score = data_df["eval_two_roc_auc"].mean()
    sns.violinplot(data=data_df, x="Variant Class", y="eval_two_roc_auc")
    plt.axhline(
        y=mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.2f}"
    )

    category_counts = data_df["Variant Class"].value_counts()
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


def pca_shap(
    shap_file_path: PathLike,
    pca_n_components: int = 100,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    aggregate: bool = True,
    img_save_path: PathLike | None = None,
) -> None:
    data_df = pd.read_parquet(shap_file_path)
    if aggregate:
        data_df = data_df.groupby("aaChanges").mean()

    print("p_is_var" in data_df.columns)
    return

    # Scale data
    shap_scores = data_df.drop(["aaChanges", "p_is_var"], axis=0).to_numpy(dtype=float)
    shap_scores = sklearn.preprocessing.StandardScaler().fit_transform(shap_scores)
    shap_scores = sklearn.decomposition.PCA(
        n_components=pca_n_components
    ).fit_transform(shap_scores)
    shap_scores = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=2,
        min_dist=umap_min_dist,
    ).fit_transform(shap_scores)

    # Graph umap
    data_df["Variant Class"] = data_df["aaChanges"].apply(variant_classification)
    colors = list(matplotlib.colors.CSS4_COLORS.keys())
    for curr_class, curr_color in zip(data_df["Variant Class"].unique(), colors):
        mask = (data_df["Variant Class"] == curr_class).to_numpy(dtype=bool)
        target_umap = shap_scores[mask]

        if not aggregate:
            curr_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                f"gray_to_{curr_color}", ["gray", curr_color]
            )
        else:
            curr_cmap = None

        colors = data_df.loc[mask, "p_is_var"] if not aggregate else None
        plt.scatter(
            target_umap[:, 0].flatten(),
            target_umap[:, 1].flatten(),
            c=colors,
            cmap=curr_cmap,
            vmax=1.0,
            vmin=0.0,
        )

    plt.title("Variant SHAP Score UMAP" + ("" if not aggregate else " (Aggregated)"))
    plt.xlabel("UMAP One")
    plt.ylabel("UMAP Two")

    if img_save_path is None:
        plt.show()
    else:
        plt.savefig(img_save_path)


if __name__ == "__main__":
    fire.Fire()
