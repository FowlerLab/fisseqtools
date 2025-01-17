from os import PathLike

import fire
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def graph_score_distribution(
    score_file_path: PathLike, img_save_path: PathLike | None = None
) -> None:
    data_df = pd.read_csv(score_file_path)
    scores = data_df["eval_one_roc_auc"].to_numpy()
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


if __name__ == "__main__":
    fire.Fire()
