from .cppwrap import _TSNErun
import numpy as np


def FItSNE(X: np.ndarray, no_dims: int=2, perplexity: float=30.0, theta: float=0.5, rand_seed: int=-1,
           max_iter: int=1000, stop_lying_iter: int=200, 
           fft_not_bh: bool=True, ann_not_vptree: bool=True, early_exag_coeff: float=12.0,
           no_momentum_during_exag: bool=False, start_late_exag_iter: int=-1, late_exag_coeff: float=-1, n_trees: int=50, search_k: int=-1) -> np.ndarray:
    """
    Wrapper around the Linderman et al. 2017 FItSNE C implementation

    Note: FItSNE with fft_not_bh=False, ann_not_vptree=False runs bhtSNE C++ implementation from Laurens van der Maaten

    Arguments
    ---------
    X: np.ndarray, shape (samples x dimensions)
        Input data.
    no_dims: int, default=2
        Dimensionality of the embedding
    perplexity: float, default=30.0
        Perplexity is used to determine the bandwidth of the Gaussian kernel in the input space
    theta: float, default=0.5
        Set to 0 for exact.  If non-zero, then will use either Barnes Hut or FIt-SNE based on `fft_not_bh`.
        If Barnes Hut, then this determines the accuracy of BH approximation.
    rand_seed: int, default=-1
        Random seed to get deterministic output
    max_iter: int, default=1000
        Number of iterations of t-SNE to run.
    stop_lying_iter: int, default=200
        When to switch off early exaggeration.
    fft_not_bh: bool, default=False
        If theta is nonzero, this determins whether to use FIt-SNE or Barnes Hut approximation. 
    ann_not_vptree: bool, default=False
        This determines whether to use aproximate (Annoy) or deterministic (vptree) nearest neighbours
    early_exag_coeff: float, default=12.0
        When to switch off early exaggeration. (>1)
    no_momentum_during_exag: bool=False
        Set to 0 to use momentum and other optimization tricks.
        1 to do plain, vanilla gradient descent (useful for testing large exaggeration coefficients)
    start_late_exag_iter: int, default=-1
        When to start late exaggeration. Set to -1 to not use late exaggeration
    late_exag_coeff: float, default=-1
        Late exaggeration coefficient. Set to -1 to not use late exaggeration.
    n_trees: int, default=50
        ANNOY parameter
    search_k: int, default=-1
        ANNOY parameter

    Returns
    -------
    Y: np.ndarray
        The embedded dataset
    """
    N, D = X.shape
    K = -1
    sigma = -30.0
    mom_switch_iter = 250
    # booleans
    skip_random_init = int(False)
    no_momentum_during_exag_i = int(no_momentum_during_exag)

    if fft_not_bh:
        nbody_algo = 2
    else:
        nbody_algo = 1

    if ann_not_vptree:
        knn_algo = 1
    else:
        knn_algo = 2

    print ("nbody_algo", nbody_algo)
    # memory allocations
    Y = np.zeros((N, no_dims), dtype="double")
    costs = np.zeros(max_iter, dtype="double")

    _TSNErun(X, N, D, Y, no_dims, perplexity, theta, rand_seed,
             skip_random_init, max_iter, stop_lying_iter, mom_switch_iter, K, sigma,
             nbody_algo, knn_algo, early_exag_coeff, costs, no_momentum_during_exag_i,
             start_late_exag_iter, late_exag_coeff, n_trees, search_k)

    return Y
