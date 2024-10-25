import enum
from typing import List, Tuple

import numpy as np
import pandas as pd


class VarTag(enum.Enum):
    TARGET_VAR = 0
    WILD_TYPE = 1
    NON_TARGET_MUTANT = 2


def prep_train_data(
    variant: str,
    data_df: pd.DataFrame,
    embeddings: np.ndarray,
    wildtype_class: str = "WT",
    mutant_classes: List[str] = ["Single Missense", "Multiple", "Nonsense"],
) -> Tuple[np.ndarray, np.ndarray]:
    target_df = data_df[data_df["geno"] == variant]
    n_target = len(target_df)
    non_target_df = data_df[data_df["geno"] != variant]
    wt_df = non_target_df[non_target_df["Variant_Class"] == wildtype_class].sample(
        n=n_target // 2
    )
    mutant_df = non_target_df[
        non_target_df["Variant_Class"].isin(mutant_classes)
    ].sample(n=n_target // 2)

    embedding_indices = np.concat(
        (
            target_df["embedding_index"].to_numpy(),
            wt_df["embedding_index"].to_numpy(),
            mutant_df["embedding_index"].to_numpy(),
        )
    )

    sample_embeddings = embeddings[embedding_indices]
    sample_labels = np.concat(
        (
            np.repeat(VarTag.TARGET_VAR, len(target_df)),
            np.repeat(VarTag.WILD_TYPE, len(wt_df)),
            np.repeat(VarTag.NON_TARGET_MUTANT, len(mutant_df)),
        )
    )

    return sample_embeddings, sample_labels
