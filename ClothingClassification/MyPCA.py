import numpy as np
from typing import Tuple


def my_PCA(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.float32]:
    """Performs PCA on X of shape [n_samples, n_features] 
    to reduce the dimensionality to `n_components` and returns
    the transformed data and the proportion of preserved variance.
    """    

    X = X - np.mean(X, axis=0)
    cov = X.T @ X / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs].real
    eigenvectors = eigenvectors[idxs].real
    components = eigenvectors[:n_components]
    prop = eigenvalues[idxs][:n_components].sum()/eigenvalues.sum()
    print("PCA: Proportion of preserved variance: {}%".format(
        round(100 * prop, 2)))
    return np.dot(X, components.T).real, prop
