import pathlib

import numpy as np
import pandas as pd
import pickle
import pytest
import sklearn.ensemble

import fisseqtools.grad_boost_select


def test_prep_test_data():
    data_df = pd.DataFrame(
        {
            "geno": ["A"] * 2 + ["B"] * 2 + ["C"] * 2,
            "Variant_Class": ["Single Missense"] * 2 + ["WT"] * 2 + ["Multiple"] * 2,
            "embedding_index": np.arange(6),
        }
    )
    embeddings = np.arange(20).reshape((20, 1))
    sample_embeddings, sample_labels = fisseqtools.grad_boost_select.prep_train_data(
        "A",
        data_df,
        embeddings,
    )

    sample_embeddings = sample_embeddings.flatten()
    assert len(sample_embeddings) == 4
    assert len(sample_labels) == 4
    assert np.array_equal(
        sample_labels[0:2],
        np.repeat(fisseqtools.grad_boost_select.VarTag.TARGET_VAR.value, 2),
    )
    assert np.array_equal(sample_embeddings[0:2], np.array([0, 1]))
    assert sample_labels[2] == fisseqtools.grad_boost_select.VarTag.WILD_TYPE.value
    assert sample_embeddings[2] in [2, 3]
    assert (
        sample_labels[3] == fisseqtools.grad_boost_select.VarTag.NON_TARGET_MUTANT.value
    )
    assert sample_embeddings[3] in [4, 5]


def test_train_model():
    classifier_hyperparams = {"n_estimators": 25, "max_depth": 1}
    data_df = pd.DataFrame(
        {
            "geno": ["variant1"] * 30 + ["other"] * 30,
            "Variant_Class": ["Single Missense"] * 30 + ["WT"] * 15 + ["Multiple"] * 15,
            "embedding_index": np.arange(60),
        }
    )
    embeddings = np.zeros((60, 1))
    # Impossible to differentiate wild type from target variant
    # Trivial to differentiate non target mutations from target variant
    embeddings[:45, :] = 1

    (
        model,
        test_auc,
        test_wt_auc,
        test_mutant_auc,
    ) = fisseqtools.grad_boost_select.train_model(
        classifier_hyperparams,
        data_df,
        embeddings,
        "variant1",
    )

    assert isinstance(model, sklearn.ensemble.GradientBoostingClassifier)
    assert test_auc == pytest.approx(2 / 3)
    assert test_wt_auc == pytest.approx(1 / 2)
    assert test_mutant_auc == pytest.approx(1)


def test_train_models(tmpdir):
    tempdir = pathlib.Path(tmpdir)
    data_df_path = tempdir / "data.csv"
    embeddings_pkl_path = tempdir / "embeddings.pkl"
    results_dir_path = tempdir / "results"
    results_dir_path.mkdir(exist_ok=True)

    pd.DataFrame(
        {
            "geno": ["variant1"] * 30 + ["other1"] * 15 + ["other2"] * 15,
            "Variant_Class": ["Single Missense"] * 30 + ["WT"] * 15 + ["Multiple"] * 15,
            "embedding_index": np.arange(60),
        }
    ).to_csv(data_df_path)
    embeddings = np.zeros((60, 1))
    # Impossible to differentiate wild type from target variant
    # Trivial to differentiate non target mutations from target variant
    embeddings[:45, :] = 1

    with open(embeddings_pkl_path, "wb") as f:
        pickle.dump(embeddings, f)

    # Run the train_models function
    fisseqtools.grad_boost_select.train_models(
        data_df_path=data_df_path,
        embeddings_pkl_path=embeddings_pkl_path,
        results_dir_path=results_dir_path,
        num_threads=1,
        max_depth=1,
        max_estimators=25,
        early_stop_iter=25,
        min_cell_count=30,
    )

    # Check classifiers
    classifiers_path = results_dir_path / "classifiers"
    assert classifiers_path.exists() and classifiers_path.is_dir()
    assert (classifiers_path / "gboost_1.pkl").is_file()

    # Check that results.csv
    results_csv_path = results_dir_path / "results.csv"
    assert results_csv_path.exists()
    results_df = pd.read_csv(results_csv_path)
    assert len(results_df) == 1
    assert results_df.iloc[0, 0] == "gboost_1.pkl"
    assert results_df.iloc[0, 1] == "variant1"
    assert results_df.iloc[0, 2] == pytest.approx(2 / 3)
    assert results_df.iloc[0, 3] == pytest.approx(1 / 2)
    assert results_df.iloc[0, 4] == pytest.approx(1)
