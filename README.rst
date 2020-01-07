FFT-accelerated Interpolation-based t-SNE (FIt-SNE)
===================================================
Introduction
------------
t-Stochastic Neighborhood Embedding ([t-SNE](https://lvdmaaten.github.io/tsne/)) is a highly successful method for dimensionality reduction and visualization of high dimensional datasets.  A popular `implementation <https://github.com/lvdmaaten/bhtsne>`_ of t-SNE uses the Barnes-Hut algorithm to approximate the gradient at each iteration of gradient descent. We accelerated this implementation as follows:

- Computation of the N-body Simulation: Instead of approximating the N-body simulation using Barnes-Hut, we interpolate onto an equispaced grid and use FFT to perform the convolution, dramatically reducing the time to compute the gradient at each iteration of gradient descent. See the `this <http://gauss.math.yale.edu/~gcl22/blog/numerics/low-rank/t-sne/2018/01/11/low-rank-kernels.html>`_ post for some intuition on how it works.
- Computation of Input Similarities: Instead of computing nearest neighbors using vantage-point trees, we approximate nearest neighbors using the `Annoy <https://github.com/spotify/annoy>`_ library. The neighbor lookups are multithreaded to take advantage of machines with multiple cores. Using "near" neighbors as opposed to strictly "nearest" neighbors is faster, but also has a smoothing effect, which can be useful for embedding some datasets (see `Linderman et al. (2017) <https://arxiv.org/abs/1711.04712>`_). If subtle detail is required (e.g. in identifying small clusters), then use vantage-point trees (which is also multithreaded in this implementation).


Check out our `paper <https://www.nature.com/articles/s41592-018-0308-4>`_ or `preprint <https://arxiv.org/abs/1712.09005>`_ for more details and some benchmarks.

Features
--------
Additionally, this implementation includes the following features:

- Early exaggeration: In `Linderman and Steinerberger (2018) <https://epubs.siam.org/doi/abs/10.1137/18M1216134>`_, we showed that appropriately choosing the early exaggeration coefficient can lead to improved embedding of swissrolls and other synthetic datasets. Early exaggeration is built into all t-SNE implementations; here we highlight its importance as a parameter.
- Late exaggeration: Increasing the exaggeration coefficient late in the optimization process can improve separation of the clusters. `Kobak and Berens (2019) <https://www.nature.com/articles/s41467-019-13056-x>`_ suggest starting late exaggeration immediately following early exaggeration.
- Initialization: Custom initialization can be provided from Python/Matlab/R. As suggested by `Kobak and Berens (2019) <https://www.nature.com/articles/s41467-019-13056-x>`_, initializing t-SNE with the first two principal components (scaled to have standard deviation 0.0001) results in an embedding which often preserves the global structure more effectively than the default random normalization. See there for other initialisation tricks.
- Variable degrees of freedom: `Kobak et al. (2019) <https://ecmlpkdd2019.org/downloads/paper/327.pdf>`_ show that decreasing the degree of freedom (df) of the t-distribution (resulting in heavier tails)  reveals fine structure that is not visible in standard t-SNE.
- All optimisation parameters can be controlled from Python. For example, `Belkina et al. (2019) <https://www.nature.com/articles/s41467-019-13055-y>`_ highlight the importance of increasing the learning rate when embedding large data sets.

Implementations
---------------
There are (at least) three options for using FIt-SNE in Python:

- **This PyPI package** (see installation instructions below), which is a Cython wrapper for `FIt-SNE <https://github.com/KlugerLab/FIt-SNE>`_ and was written by `Gioele La Manno <https://twitter.com/GioeleLaManno>`_. This package is not directly updated; rather, the C++ code from `FIt-SNE <https://github.com/KlugerLab/FIt-SNE>`_ is ported here occasionally. The current version of the C++ code corresponds to FIt-SNE 1.1.0 (commit 714e12e).
- The Python wrapper available from the `FIt-SNE <https://github.com/KlugerLab/FIt-SNE>`_ Github. It is not on PyPI, but rather wraps the FIt-SNE binary.
- `OpenTSNE <https://github.com/pavlin-policar/openTSNE/>`_, which is a pure Python implementation of FIt-SNE, also available on PyPI.

Installation
------------
The only prerequisite is `FFTW <http://www.fftw.org/>`__. FFTW and fitsne can be installed as follows:

.. code:: bash

   conda config --add channels conda-forge #if not already in your channels. Needed for fftw.
   conda install cython numpy fftw
   pip install fitsne

And you're good to go!

Bug reports, feature requests, etc.
-------------------------------------
If you have any problems with this package, please open an issue on the Github `repository <https://github.com/KlugerLab/pyFIt-SNE>`__.

References
----------

If you use our software, please cite:

George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. Nature Methods.  (`link <https://www.nature.com/articles/s41592-018-0308-4>`__)

Our implementation is derived from the Barnes-Hut implementation:

Laurens van der Maaten (2014). Accelerating t-SNE using tree-based
algorithms. Journal of Machine Learning Research, 15(1):3221–3245.
(`link <https://dl.acm.org/citation.cfm?id=2627435.2697068>`__)


