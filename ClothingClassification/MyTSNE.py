import numpy as np
from tqdm import tqdm


def get_pairwise_dist(X: np.ndarray) -> np.ndarray:
    """Computes pairwise distances**2 between rows of `X`."""   

    Xs = (X**2).sum(axis=1, keepdims=True)
    D = Xs + Xs.T - 2 * X @ X.T
    return D


def get_P(X: np.ndarray, sigma: int) -> np.ndarray:
    """Computes distribution P for x's."""
    
    n = X.shape[0]
    D = get_pairwise_dist(X)
    P = np.exp(-D/2/sigma**2)
    P = P / (np.sum(P, axis=1, keepdims=True) - 1)
    P = (P + P.T) / (2 * n)
    for i in range(n):
        P[i, i] = 0
    return P


def get_Q(Y: np.ndarray) -> np.ndarray:
    """Computes distribution Q for y's."""  

    n = Y.shape[0]
    D = get_pairwise_dist(Y)
    Q = 1 / (1 + D)
    Q = Q / (np.sum(Q)-n)
    for i in range(n):
        Q[i, i] = 0
    return Q


def get_gradient(Y: np.ndarray, P: np.ndarray, Q: np.ndarray, n_components: int = 2):
    """Computes the gradient of the loss function KL(P || Q) w.r.t. Y."""    

    n = Y.shape[0]
    dY = np.zeros((n, n_components))
    D = get_pairwise_dist(Y)
    K = 4 * np.divide((P - Q), (1 + D))
    for i in range(n):
        for j in range(n):
            dY[i] += K[i,j] * (Y[i] - Y[j])
    return dY


def my_tSNE(X: np.ndarray, n_components: int = 2, sigma: float = 30, 
            max_iter: int = 10000, lr: float = 10, eps: float = 1e-6) -> np.ndarray:
    """Performs t-SNE on X of shape [n_samples, n_features] 
    to reduce the dimensionality to `n_components`."""    

    # Initialize y's
    n = X.shape[0]
    Y = np.random.randn(n, n_components)
    # Compute distribution P for x's
    P = get_P(X, sigma)
    # Gradient descent
    prev_loss = 0
    with tqdm(range(max_iter), desc='Optimizing t-SNE') as pbar:
        thresh = max_iter // 10  # Reduce learning rate after this many iterations
        for i in pbar:
            if i % thresh == 0:
                lr /= 2
            Q = get_Q(Y)  # Compute distribution Q for y's
            dY = get_gradient(Y, P, Q)
            Y -= lr * dY
            for i in range(n):
                Q[i,i] = 1
            loss = -np.sum(P * np.log(Q))
            if abs(loss - prev_loss) < eps:
                break
            prev_loss = loss
            if i % 50 == 0:
                pbar.set_postfix({"loss": f"{loss:.3f}"})
    return Y
