import numpy as np
from scipy.sparse.linalg import svds


def eofs(data, num_modes):
    """
    Compute Empirical Orthogonal Functions using truncated SVD.

    Parameters
    ----------
    data : ndarray, shape (M, N)
        Snapshots in columns (space x time)
    num_modes : int
        Number of EOF modes to retain

    Returns
    -------
    lambda_vals : ndarray, shape (num_modes,)
        Eigenvalues (unnormalized, descending order)
    u : ndarray, shape (M, num_modes)
        Spatial EOFs (eigenvectors as columns)
    coeff : ndarray, shape (num_modes, N)
        Time coefficients
    cumul_approx : ndarray, shape (num_modes, M, N)
        Cumulative approximations using 1..k modes
    """

    # ------------------
    # Setup
    # ------------------
    M, N = data.shape

    # Remove mean (along time dimension)
    data = data - data.mean(axis=1, keepdims=True)

    # ------------------
    # Compute EOFs by SVD
    # ------------------
    # MATLAB: svds(data/sqrt(N-1), num_modes)
    U, s, Vt = svds(data / np.sqrt(N - 1), k=num_modes)

    # svds does NOT guarantee descending order → sort manually
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]

    # Eigenvalues
    lambda_vals = s**2

    # ------------------
    # Time series coefficients
    # ------------------
    # MATLAB: coeff = u' * data
    coeff = U.T @ data

    # ------------------
    # Cumulative approximations
    # ------------------
    cumul_approx = np.zeros((num_modes, M, N))

    for kk in range(num_modes):
        # sum_to_kk = np.zeros((M, N))
        # for ii in range(kk + 1):
        #     # MATLAB: bsxfun(@times, u(:,ii), coeff(ii,:))
        #     sum_to_kk += U[:, ii][:, None] * coeff[ii, :][None, :]
        # cumul_approx[kk, :, :] = sum_to_kk

        cumul_approx[kk] = U[:, :kk+1] @ coeff[:kk+1, :]

    return lambda_vals, U, coeff, cumul_approx