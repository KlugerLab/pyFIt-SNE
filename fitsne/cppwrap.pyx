import cython
from libcpp cimport bool

cdef extern from "src/tsne.h":
    cdef cppclass TSNE:
        TSNE() except +
        int run(double *X, int N, int D, double *Y, int no_dims, double perplexity, double theta, int rand_seed,
            bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter, 
            double momentum, double final_momentum, double learning_rate, int K, double sigma,
            int nbody_algorithm, int knn_algo, double early_exag_coeff, double *costs,
            bool no_momentum_during_exag, int start_late_exag_iter, double late_exag_coeff, int n_trees, int search_k,
            int nterms, double intervals_per_integer, int min_num_intervals, unsigned int nthreads, int load_affinities,
            int perplexity_list_length, double *perplexity_list, double df, double max_step_norm );


def _TSNErun(double[:, ::1] X, int N, int D, double[:, ::1] Y, int no_dims, double perplexity, double theta, int rand_seed,
             bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter, 
             double momentum, double final_momentum, double learning_rate, int K, double sigma,
             int nbody_algo, int knn_algo, double early_exag_coeff, double[::1] costs,
             int no_momentum_during_exag, int start_late_exag_iter, double late_exag_coeff, int n_trees, int search_k,
             int nterms, double intervals_per_integer, int min_num_intervals,  unsigned int nthreads, int load_affinities, double[::1] perplexity_list, double df, double max_step_norm):
    tsne_obj = new TSNE()

    cdef:
        bool skip_random_init_b = <bool>skip_random_init
        bool no_momentum_during_exag_b = <bool>no_momentum_during_exag
        double initialError
    try:
        tsne_obj.run(&X[0, 0], N, D, &Y[0, 0], no_dims, perplexity, theta, rand_seed,
                     skip_random_init_b, max_iter, stop_lying_iter, mom_switch_iter,
                     momentum,final_momentum,learning_rate, K, sigma,
                     nbody_algo, knn_algo, early_exag_coeff,  &costs[0],
                     no_momentum_during_exag_b, start_late_exag_iter, late_exag_coeff, n_trees, search_k,
                    nterms, intervals_per_integer, min_num_intervals, nthreads,load_affinities, len(perplexity_list),&perplexity_list[0], df, max_step_norm)
    finally:
        del tsne_obj
