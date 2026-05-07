from pathlib import Path

import numpy as np


def load_spiral_matrix(file_path):
    data_file = Path(file_path)
    if not data_file.is_absolute():
        data_file = Path(__file__).resolve().parent / data_file

    return np.loadtxt(data_file, delimiter=",")


def create_train_test_split(matrix):
    working_matrix = matrix.copy()
    working_matrix = np.hstack((-np.ones((working_matrix.shape[0], 1)), working_matrix))

    np.random.shuffle(working_matrix)

    split_point = int(0.8 * len(working_matrix))
    train_matrix = working_matrix[:split_point]
    test_matrix = working_matrix[split_point:]
    return train_matrix, test_matrix


def normalize_train_test(train_matrix, test_matrix):
    train_matrix_norm = train_matrix.copy()
    test_matrix_norm = test_matrix.copy()

    train_features = train_matrix[:, 1:3]
    feature_min = train_features.min(axis=0)
    feature_max = train_features.max(axis=0)
    feature_range = np.where(feature_max - feature_min == 0, 1, feature_max - feature_min)

    train_matrix_norm[:, 1:3] = 2 * ((train_matrix[:, 1:3] - feature_min) / feature_range) - 1
    test_matrix_norm[:, 1:3] = 2 * ((test_matrix[:, 1:3] - feature_min) / feature_range) - 1

    return train_matrix_norm, test_matrix_norm
