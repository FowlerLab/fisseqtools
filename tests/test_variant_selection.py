import numpy as np
import pandas as pd

import fisseqtools.variant_selection


def test_prep_test_data():
    data_df = pd.DataFrame(
        {
            "geno": ["A"] * 2 + ["B"] * 2 + ["C"] * 2,
            "Variant_Class": ["Single Missense"] * 2 + ["WT"] * 2 + ["Multiple"] * 2,
            "embedding_index": np.arange(6),
        }
    )
    embeddings = np.arange(20).reshape((20, 1))
    sample_embeddings, sample_labels = fisseqtools.variant_selection.prep_train_data(
        "A",
        data_df,
        embeddings,
    )

    sample_embeddings = sample_embeddings.flatten()
    assert len(sample_embeddings) == 4
    assert len(sample_labels) == 4
    assert np.array_equal(
        sample_labels[0:2],
        np.repeat(fisseqtools.variant_selection.VarTag.TARGET_VAR, 2),
    )
    assert np.array_equal(sample_embeddings[0:2], np.array([0, 1]))
    assert sample_labels[2] == fisseqtools.variant_selection.VarTag.WILD_TYPE
    assert sample_embeddings[2] in [2, 3]
    assert sample_labels[3] == fisseqtools.variant_selection.VarTag.NON_TARGET_MUTANT
    assert sample_embeddings[3] in [4, 5]
