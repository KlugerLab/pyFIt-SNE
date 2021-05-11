from .cppwrap import _TSNErun
import numpy as np


def FItSNE(X: np.ndarray, no_dims: int=2, perplexity: float=30.0, 
           sigma: float=-30.0, K: int=-1, initialization = 'pca',
           load_affinities = None,perplexity_list=None,
           theta: float=0.5, rand_seed: int=-1,
           max_iter: int=750, stop_early_exag_iter: int=250, 
           fft_not_bh: bool=True, ann_not_vptree: bool=True, early_exag_coeff: float=12.0,
           no_momentum_during_exag: bool=False, start_late_exag_iter: int=-1, late_exag_coeff: float=-1, 
           mom_switch_iter: int=250,momentum: float=0.5, final_momentum: float=0.8, learning_rate='auto', max_step_norm: float=5, n_trees: int=50, search_k: int=None,
           nterms: int=3, intervals_per_integer: float=1, min_num_intervals: int=50, nthreads:  int=0, df: float=1.0) -> np.ndarray:
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
    initialization: 'random', 'pca', or numpy array
         N x no_dims array to intialize the solution. Default: 'pca'.
    load_affinities: {'load', 'save', None}
        If 'save', input similarities (p_ij) are saved into a file. If 'load', 
        they are loaded from a file and not recomputed. If None, they are not
        saved and not loaded. Default is None.
    perplexity_list: list
        A list of perplexities to used as a perplexity combination. Input 
        affinities are computed with each perplexity on the list and then
        averaged. Default is None.
    theta: float, default=0.5
        Set to 0 for exact.  If non-zero, then will use either Barnes Hut or FIt-SNE based on `fft_not_bh`.
        If Barnes Hut, then this determines the accuracy of BH approximation.
    rand_seed: int, default=-1
        Random seed to get deterministic output
    max_iter: int, default=750
        Number of iterations of t-SNE to run.
    stop_early_exag_iter: int, default=250
        When to switch off early exaggeration.
    fft_not_bh: bool, default=False
        If theta is nonzero, this determins whether to use FIt-SNE or Barnes Hut approximation. 
    ann_not_vptree: bool, default=False
        This determines whether to use aproximate (Annoy) or deterministic (vptree) nearest neighbours
    early_exag_coeff: float, default=12.0
        When to switch off early exaggeration. (>1)
    mom_switch_iter: int, default=250
        When to switch momentum
    momentum: float, default=0.5
        Size of momentum
    final_momentum: float, default=0.8
        Final momentum
    learning_rate: double or 'auto'
        Learning rate. Default 'auto'; it sets learning rate to 
        N/early_exag_coeff where N is the sample size, or to 200 if 
        N/early_exag_coeff < 200.
    max_step_norm: double or -1 (default: 5)
        Maximum distance that a point is allowed to move on one iteration. 
        Larger steps are clipped to this value. This prevents possible
        instabilities during gradient descent. Set to -1 to switch it off.
    no_momentum_during_exag: bool=False
        Set to 0 to use momentum and other optimization tricks.
        1 to do plain, vanilla gradient descent (useful for testing large exaggeration coefficients)
    start_late_exag_iter: int, default=-1
        When to start late exaggeration. Set to -1 to not use late exaggeration
    late_exag_coeff: float, default=-1
        Late exaggeration coefficient. Set to -1 to not use late exaggeration.
    n_trees: int, default=50
        ANNOY parameter
    search_k: int
        When using Annoy, the number of nodes to inspect during search. Default 
        is 3*perplexity*n_trees (or K*n_trees when using fixed sigma).
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
   df: float, default=0 
       Degree of freedom of t-distribution, must be greater than 0.
       Values smaller than 1 correspond to heavier tails, which can often 
       resolve substructure in the embedding. See Kobak et al. (2019) for
       details. 

       

    Returns
    -------
    Y: np.ndarray
        The embedded dataset
    """
    N, D = X.shape
    # booleans
    no_momentum_during_exag_i = int(no_momentum_during_exag)

    if learning_rate == "auto":
        learning_rate = np.max((200, X.shape[0] / early_exag_coeff))

#     if initialization is not None:
#        skip_random_init = int(True)
#        Y = initialization
#    else:
#        skip_random_init = int(False)
#        Y = np.zeros((N, no_dims), dtype="double")

    if isinstance(initialization, str) and initialization == "pca":
        from sklearn.decomposition import PCA

        solver = "arpack" if X.shape[1] > no_dims else "auto"
        pca = PCA(
            n_components=no_dims,
            svd_solver=solver,
            random_state=rand_seed if rand_seed != -1 else None,
        )
        Y = pca.fit_transform(X)
        Y /= np.std(Y[:, 0])
        Y *= 0.0001
        Y  = Y.copy(order='C')
        skip_random_init = int(True)
    elif(isinstance(initialization, str) and initialization == "random"):
        skip_random_init = int(False)
        Y = np.zeros((N, no_dims), dtype="double")
    else:
        skip_random_init = int(True)
        Y = initialization.copy(order='C')
        
    if perplexity_list is None:
        perplexity_list = np.zeros(1, dtype="double")
    else:
        perplexity = 0  # C++ requires perplexity=0 in order to use perplexity_list
        perplexity_list = np.array(perplexity_list, dtype = "double")

    if search_k is None:
        if perplexity > 0:
            search_k = 3 * perplexity * n_trees
        elif perplexity == 0:
            search_k = 3 * np.max(perplexity_list) * n_trees
        else:
            search_k = K * n_trees
        
    if load_affinities == "load":
        load_affinities = 1
    elif load_affinities == "save":
        load_affinities = 2
    else:
        load_affinities = 0




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
             skip_random_init, max_iter, stop_early_exag_iter, mom_switch_iter, 
             momentum,final_momentum,learning_rate,K, sigma,
             nbody_algo, knn_algo, early_exag_coeff, costs, no_momentum_during_exag_i,
             start_late_exag_iter, late_exag_coeff, n_trees, search_k, 
             nterms, intervals_per_integer, min_num_intervals,nthreads, load_affinities,perplexity_list, df, max_step_norm)

    return Y
