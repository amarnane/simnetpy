"""
Extension of spectral cluster class developed in https://pypi.org/project/spectralcluster/
Allow the passing of adjacency matrices. Adjusted laplacian and eigengap parameters to accept strings.
Removed a lot of functionality relating to constraint options and refinement preprocessing. 
"""

import spectralcluster as sc
import numpy as np
import warnings

from ..similarity import pairwise_sim

LTYPES = {
    'a':sc.LaplacianType.Affinity,
    'l':sc.LaplacianType.Unnormalized,
    'lrw':sc.LaplacianType.RandomWalk,
    'lsym':sc.LaplacianType.GraphCut,
}

EIGENGAPCOMP = {
    'ratio':sc.utils.EigenGapType.Ratio,
    'normdiff':sc.utils.EigenGapType.NormalizedDiff,
}


class Spectral(sc.SpectralClusterer):
    def __init__(self, 
                min_clusters=2,
                max_clusters=10, 
                laplacian_type='lrw', 
                stop_eigenvalue=1e-2, 
                custom_dist="cosine", 
                eigengap_type='ratio', **kwds):
        ltype = LTYPES[laplacian_type.lower()] # lookup laplacian type class, A/L/Lrw/Lsym (all get lower cased)
        egaptype = EIGENGAPCOMP[eigengap_type] # lookup eigengap comp type
        
        super().__init__(min_clusters=min_clusters,
                max_clusters=max_clusters, 
                laplacian_type=ltype, 
                stop_eigenvalue=stop_eigenvalue, 
                custom_dist=custom_dist, 
                eigengap_type=egaptype, **kwds)
    
    def predict_from_adj(self, A, laplacian_type=None):
        if laplacian_type is not None:
            self.laplacian_type = LTYPES[laplacian_type.lower()]

        n_samples = A.shape[0]

        constraint_matrix=None # ignore constraint for now. not clear what use is.
        eigenvectors, n_clusters, _ = self._compute_eigenvectors_ncluster(
            A, constraint_matrix)

        if self.min_clusters is not None:
            n_clusters = max(n_clusters, self.min_clusters)

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :n_clusters]

        if self.row_wise_renorm:
            # Perform row wise re-normalization.
            rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
            spectral_embeddings = spectral_embeddings / np.reshape(
                rows_norm, (n_samples, 1))

        # Run clustering algorithm on spectral embeddings. This defaults
        # to customized K-means.
        # kmeans is raising future warning. rather than rewrite entire class. I added a simple warning filter.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            labels = self.post_eigen_cluster_function(
                spectral_embeddings=spectral_embeddings,
                n_clusters=n_clusters,
                custom_dist=self.custom_dist,
                max_iter=self.max_iter)
        return labels


    def predict_from_aff(self, X=None, S=None, metric='euclidean', norm=True):
        """Perform spectral clustering on an affinity matrix. 
        Note affinity matrix should have form where larger values indicate higher similarity i.e. opposite of distance 
        Args:
            X:      embedding/feature matrix n x d np.ndarray
            S:    pairwise affinity (Similarity) matrix n x n np.ndarray. 
            metric: metric to be used in pairwise distance
            norm:   wether to normalise Aff to be 0 mean 1 std
        Returns:
            labels: numpy array of shape (n_samples,)
        """
        self.laplacian_type = LTYPES['a'] # set laplacian type as affinity
        if (X is None) and (S is None):
            raise ValueError('One of X or S must be specified. Note if both are specified then S is used.')

        if S is None:
            affinity = self.affinity_matrix(X, metric=metric, norm=norm)
        else:
            affinity = S

        num_embeddings = affinity.shape[0]
            
        # # Compute affinity matrix.
        # affinity = self.affinity_function(embeddings)
        constraint_matrix = None

        eigenvectors, n_clusters, _ = self._compute_eigenvectors_ncluster(
                affinity, constraint_matrix)

        if self.min_clusters is not None:
            n_clusters = max(n_clusters, self.min_clusters)

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :n_clusters]

        if self.row_wise_renorm:
            # Perform row wise re-normalization.
            rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
            spectral_embeddings = spectral_embeddings / np.reshape(
                rows_norm, (num_embeddings, 1))

        # Run clustering algorithm on spectral embeddings. This defaults
        # to a customized implementation of K-means. However, the implementation was created on old sklearn version.
        # -> kmeans raises a Future Warning for n_init parameter.  Rather than rewrite entire class. I added a simple warning filter.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            labels = self.post_eigen_cluster_function(
                spectral_embeddings=spectral_embeddings,
                n_clusters=n_clusters,
                custom_dist=self.custom_dist,
                max_iter=self.max_iter)
        return labels

    @staticmethod
    def affinity_matrix(X, metric='euclidean', norm=True):
        S = pairwise_sim(X, method=metric, norm=norm)
        S = -S  # want higher values to be more similar
        return S
