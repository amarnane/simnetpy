import numpy as np
import pandas as pd
import igraph as ig

from scipy.spatial.distance import squareform
from collections import defaultdict

from ..graph import Igraph


def mat2graph(A):
    # print(A.sum())
    g = Igraph().Adjacency(A, mode='undirected')
    return g

def threshold_adj(D, t):
    """Threshold dissimilarity matrix using quantile of values. Assumes distance. Edges retained are values below
    smallest t%.
    
    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        t (float): 0 to 1 top 100*t% of edges to keep. 0.01 means top 1% most similar connections.

    Returns:
        np.ndarray: Adjacency matrix of 0 and 1s
    """
    A = np.zeros(D.shape)
    # d = np.triu(D).flatten()
    np.fill_diagonal(D,0) # coversion to squareform requires 0 diagonals
    d = squareform(D)
    t = np.quantile(d, t)
    A[D < t] = 1
    return A

def knn_adj(D,K):
    """Create a network from a dissimilarity matrix by finding top K most similar neighbours for each node. 
    Note: uses brute force algorithm. Checks all possible values. Slow for large matrices

    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        K (int): Number of Neighbours to find for each individual

    Returns:
        np.ndarray: Adjacency matrix of 0 and 1s
    """
    assert isinstance(K, (int, np.integer)), "K must be an integer"

    # Note D is distance matrix not affinity
    # lower values => more similar.
    # output from pairwise_sim has diagonals with minimum value 
    np.fill_diagonal(D, D.max()) # don't want to include diagonals

    A = np.zeros(D.shape)
    # find K nearest neighbours for each individual 
    nn = np.argsort(D, axis=1)[:,:K]
    # add edge (set value = 1) for each nearest neighbour
    np.put_along_axis(A,nn, 1, axis=1) 
    # make symmetric
    A = A.T + A
    # if two individuals both have each other as nn then value will be 2. set == 1
    A[A>1] = 1
    return A

    
def combined_adj(D, K, t):
    """Create a network from a dissimilarity matrix through a mixture of KNN and global
    threshold.

    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        K (int): Number of Neighbours to find for each individual
        t (float): 0 to 1 top 100*t% of edges to keep. 0.01 means top 1% most similar connections.

    Returns:
        np.ndarray: Adjacency matrix of 0 and 1s
    """
    A_k = knn_adj(D, K)

    A_t = threshold_adj(D, t)
    A = A_k + A_t
    A[A>1] = 1
    return A

def threshold_graph(D,t):
    """Threshold dissimilarity matrix using quantile of values and create a igraph network. Assumes distance. Edges retained are values below
    smallest t%.
    
    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        t (float): 0 to 1 top 100*t% of edges to keep. 0.01 means top 1% most similar connections.

    Returns:
        ig.Graph: Graph created from thresholding connections.
    """
    A = threshold_adj(D,t)
    g = mat2graph(A)
    return g



def neighbour_func1d(x, k=10, func='mean'):
    funcs = {'mean':np.mean, 'median':np.median, 'std':np.std}
    f = funcs[func]
    nn = x.argsort()[:k]
    return f(x[nn])

def neighbour_stat(D, k=10, axis=0, stat='mean'):
    return np.apply_along_axis(neighbour_func1d, arr=D, axis=axis, **{'k':k,'func':stat})


def nn_distribution(D, K, statNN=10, stat='mean', Kquantile=1.0, mapping='linear'):
    """

    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        K (int): Control number of Neighbours in neighbour distribution. Coupled with Kquantile. 
            e.g. K=5, Kquantile=0.5 means 5 will be mean of distribution. kquantile=1.0 mean 5 will be max. 
        statNN (int, optional): Number of neighbours in stat calc. Defaults to 10.
        stat (str, optional): Stat to calculate from neighest neighbours, one of mean, median, std. 
                            Defaults to 'mean'.
        Kquantile (float, optional): Quantile of stat dist to map K to. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    stat = stat.lower()
    assert stat in ['mean', 'median', 'std'], 'Must be one of mean, median, std'
    assert Kquantile <= 1.0 and Kquantile>=0, "Must be between 0 and 1"
    assert mapping in ['linear', 'log'], "mapping must be one of [linear, log]"
    S = neighbour_stat(D, k=statNN, stat=stat) # find average similarity amongst statNN closest neighbours
    V = - (S - S.min())/(S.max()-S.min()) + 1 # normalise to 0, 1. Note: close to 1 means larger number of similar neighbours

    # We map the distribution from to 0,Kmax so that the number of neighbours each node is assigned goes from 0, Kmax
    Kq = np.quantile(V, Kquantile)
    Kmax = int(K/Kq) 
    if mapping == 'linear':
        V = Kmax*V # normalise to 0,K
        NN = np.digitize(V, bins=np.arange(Kmax))
    elif mapping == 'log':
        upper = np.log(Kmax+1)
        V = upper*V # normalise to 0,log(Kmax+1)
        V = np.exp(V) # V now in [1, Kmax+1]
        NN = np.digitize(V-1, bins=np.arange(Kmax)) # digitize maps x in [0-1] to 1 
                                        #i.e. rounds up so need -1 to get correct range
    return NN

def skewed_knn_adj(D, K, statNN=10, stat='mean', Kquantile=1.0):
    """Create a network from a dissimilarity matrix through a mixture of KNN and global
    threshold.

    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        K (int): Control number of Neighbours in neighbour distribution. Coupled with Kquantile. 
            e.g. K=5, Kquantile=0.5 means 5 will be mean of distribution. kquantile=1.0 mean 5 will be max. 
        statNN (int, optional): Number of neighbours in stat calc. Defaults to 10.
        stat (str, optional): Stat to calculate from neighest neighbours, one of mean, median, std. 
                            Defaults to 'mean'.
        Kquantile (float, optional): Quantile of stat dist to map K to. Defaults to 1.0.

    Returns:
        np.ndarray: Adjacency matrix of 0 and 1s
    """
    assert isinstance(K,int), "K must be an integer"
    
    np.fill_diagonal(D,D.max())
    NN = nn_distribution(D, K, statNN=statNN, stat=stat, Kquantile=Kquantile, mapping='linear')
    D_sorted = np.argsort(D, axis=1)
    A = np.zeros(D.shape)
    for i, nn in enumerate(NN):
        idx = D_sorted[i,:nn]
        A[i, idx] = 1

    A = A.T + A
    A[A>1] = 1
    return A


def log_skewed_knn_adj(D, K, statNN=10, stat='mean', Kquantile=1.0):
    """Create a network from a dissimilarity matrix through a mixture of KNN and global
    threshold.

    Args:
        D (np.ndarray): nxn dissimilarity matrix. smaller values => more similar.
        K (int): Control number of Neighbours in neighbour distribution. Coupled with Kquantile. 
            e.g. K=5, Kquantile=0.5 means 5 will be mean of distribution. kquantile=1.0 mean 5 will be max. 
        statNN (int, optional): Number of neighbours in stat calc. Defaults to 10.
        stat (str, optional): Stat to calculate from neighest neighbours, one of mean, median, std. 
                            Defaults to 'mean'.
        Kquantile (float, optional): Quantile of stat dist to map K to. Defaults to 1.0.

    Returns:
        np.ndarray: Adjacency matrix of 0 and 1s
    """
    assert isinstance(K,int), "K must be an integer"
    
    np.fill_diagonal(D,D.max())
    NN = nn_distribution(D, K, statNN=statNN, stat=stat, Kquantile=Kquantile, mapping='log')
    D_sorted = np.argsort(D, axis=1)
    A = np.zeros(D.shape)
    for i, nn in enumerate(NN):
        idx = D_sorted[i,:nn]
        A[i, idx] = 1

    A = A.T + A
    A[A>1] = 1
    return A

def sparsify_sim_matrix(D, method='knn', **kwargs):
    """
    function to sparsify dissimilarity matrix into adjacency 

    Args:
        D (np.ndarray): nxn Dissimilarity matrix. Smaller => more similar
        method (str, optional): method to use to sparsify matrix. 
                    one of [knn, threshold, combined, skewed_knn]. Defaults to 'knn'.
        **kwargs: keyword arguments for sparsifying functions

    Returns:
        np.ndarray: nxn symmetric Adjacency matrix of 0s and 1s
    """
    fsparser = {'knn': knn_adj, 'threshold':threshold_adj, 
        'combined':combined_adj, 'skewed_knn':skewed_knn_adj, 
        'log_skewed_knn':log_skewed_knn_adj}
    A = fsparser[method](D, **kwargs)
    return A

def network_from_sim_mat(D, method='knn', **kwargs):
    """
    function to sparsify dissimilarity matrix into adjacency 

    Args:
        D (np.ndarray): nxn Dissimilarity matrix. Smaller => more similar
        method (str, optional): method to use to sparsify matrix. 
                    one of [knn, threshold, combined, skewed_knn]. Defaults to 'knn'.
        **kwargs: keyword arguments for sparsifying functions

    Returns:
        ig.Graph: Graph created from similarity matrix
    """
    A = sparsify_sim_matrix(D, method=method, **kwargs)
    g = mat2graph(A)
    return g

