FFT-accelerated Interpolation-based t-SNE (FIt-SNE)
===================================================

Introduction
------------

t-Stochastic Neighborhood Embedding
(`t-SNE <https://lvdmaaten.github.io/tsne/>`__) is a highly successful
method for dimensionality reduction and visualization of high
dimensional datasets. A popular
`implementation <https://github.com/lvdmaaten/bhtsne>`__ of t-SNE uses
the Barnes-Hut algorithm to approximate the gradient at each iteration
of gradient descent. We modified this implementation as follows:

-  Computation of Input Similiarities: Instead of computing nearest
   neighbors using vantage-point trees, we approximate nearest neighbors
   using the `Annoy <https://github.com/spotify/annoy>`__ library. The
   neighbor lookups are multithreaded to take advantage of machines with
   multiple cores. Using "near" neighbors as opposed to strictly
   "nearest" neighbors is faster, but also has a smoothing effect, which
   can be useful for embedding some datasets (see `Linderman et al.
   (2017) <https://arxiv.org/abs/1711.04712>`__).
-  Computation of the N-body Simulation: Instead of approximating the
   N-body simulation using Barnes-Hut, we interpolate onto an equispaced
   grid and use FFT to perform the convolution, dramatically reducing
   the time to compute the gradient at each iteration of gradient
   descent.
-  Early exaggeration: In `Linderman and Steinerberger
   (2017) <https://arxiv.org/abs/1706.02582>`__, we showed that
   appropriately choosing the early exaggeration coefficient can lead to
   improved embedding of swissrolls and other synthetic datase ts
-  Late exaggeration: By increasing the exaggeration coefficient late in
   the optimization process (e.g. after 800 of 1000 iterations) can
   improve separation of the clusters

Check out our `preprint <https://arxiv.org/abs/1712.09005>`__ for more
details and some benchmarks.

This PyPI package is a Cython wrapper for `FIt-SNE <https://github.com/KlugerLab/FIt-SNE>`_
and was written by `Gioele La Manno <https://twitter.com/GioeleLaManno>`_.

Installation
------------
The only prerequisite is `FFTW <http://www.fftw.org/>`__. FFTW and fitsne can be installed as follows:

.. code:: bash
   
   conda config --add channels conda-forge #if not already in your channels. Needed for fftw.
   conda install cython numpy fftw  
   pip install fitsne

And you're good to go!

References
----------

If you use our software, please cite:

George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan
Steinerberger, Yuval Kluger. (2017). Efficient Algorithms for
t-distributed Stochastic Neighborhood Embedding. (2017)
*arXiv:1712.09005* (`link <https://arxiv.org/abs/1712.09005>`__)

Our implementation is derived from the Barnes-Hut implementation:

Laurens van der Maaten (2014). Accelerating t-SNE using tree-based
algorithms. Journal of Machine Learning Research, 15(1):3221–3245.
(`link <https://dl.acm.org/citation.cfm?id=2627435.2697068>`__)
