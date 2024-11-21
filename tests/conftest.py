import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def metrics_sample_data():
    np.random.seed(0)
    x_test = np.random.rand(20, 5)
    y_test = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    label_encoder = LabelEncoder().fit([0, 1, 2])
    return x_test, y_test, y_pred, label_encoder
