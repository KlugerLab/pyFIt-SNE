from .cppwrap import _TSNErun
import numpy as np


def FItSNE(X: np.ndarray, no_dims: int=2, perplexity: float=30.0, 
           sigma: float=-30.0, K: int=-1, initialization: np.ndarray=None,
           theta: float=0.5, rand_seed: int=-1,
           max_iter: int=1000, stop_lying_iter: int=200, 
           fft_not_bh: bool=True, ann_not_vptree: bool=True, early_exag_coeff: float=12.0,
           no_momentum_during_exag: bool=False, start_late_exag_iter: int=-1, late_exag_coeff: float=-1, n_trees: int=50, search_k: int=-1,
           nterms: int=3, intervals_per_integer: float=1, min_num_intervals: int=50,load_affinities: int=0, nthreads:  int=0) -> np.ndarray:
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
        Perplexity is used to determine the bandwidth of the Gaussian kernel in the input space. Set to -1 if using fixed sigma/K (see below)
    sigma: float, default=-30
        Fixed bandwidth of Gaussian kernel to use in lieu of the perplexity-based adaptive kernel width typically used in t-SNE
    K: int, default=-1
        Number of nearest neighbors to get when using fixed sigma in lieu of perplexity-based adaptive kernel width typically used in t-SNE
    initialization: np.ndarray, shape (samples x no_dims), default=None
        Initialization of the embedded points to use in lieu of the random initialization typically used in t-SNE
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
    nterms: int, default=3 
        If using FIt-SNE, this is the number of interpolation points per
        sub-interval
   intervals_per_integer: float, default=1 
        See min_num_intervals
   min_num_intervals: int, default=50
        Let maxloc = ceil(max(max(X))) and minloc = floor(min(min(X))). i.e.
        the points are in a [minloc]^no_dims by [maxloc]^no_dims
        interval/square.  The number of intervals in each dimension is either
        min_num_intervals or ceil((maxloc -minloc)/opts.intervals_per_integer), 
        whichever is larger.  opts.min_num_intervals must be an integer >0, and
        opts.intervals_per_integer must be >0.
   nthreads: unsigned int, default=0 
       Number of threads to be used in computation of input similarities (both
       for vptrees and ann). 0 uses the maximum number of threads supported by
       the hardware.

    Returns
    -------
    Y: np.ndarray
        The embedded dataset
    """
    N, D = X.shape
    mom_switch_iter = 250
    # booleans
    no_momentum_during_exag_i = int(no_momentum_during_exag)


    if initialization is not None:
        skip_random_init = int(True)
        Y = initialization
    else:
        skip_random_init = int(False)
        Y = np.zeros((N, no_dims), dtype="double")

    if fft_not_bh:
        nbody_algo = 2
    else:
        nbody_algo = 1

    if ann_not_vptree:
        knn_algo = 1
    else:
        knn_algo = 2

    # memory allocations
    costs = np.zeros(max_iter, dtype="double")

    _TSNErun(X, N, D, Y, no_dims, perplexity, theta, rand_seed,
             skip_random_init, max_iter, stop_lying_iter, mom_switch_iter, K, sigma,
             nbody_algo, knn_algo, early_exag_coeff, costs, no_momentum_during_exag_i,
             start_late_exag_iter, late_exag_coeff, n_trees, search_k, 
             nterms, intervals_per_integer, min_num_intervals,nthreads)

    return Y
