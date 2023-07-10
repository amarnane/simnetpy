import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy import stats

import snf

from .. import utils

# pdist calcs pairwise distances
# ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, 
# ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
# ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
# some of options available

def chi_square(pu, pv):
    """Chi-square distance or histogram distance.
    d(u,v) = 1/2 \sum_i=0^N (p(u_i) - p(v_i))**2/(p(u_i)+p(v_i))

    Args:
        pu (np.array): vector of probabilities of observing elements of u
        pv (np.array): vector of probabilities of observing elements of u

    Returns:
        float: distance between u and v
    """
    x = pu + pv
    x[x==0] = 1 # if u and v can't be negative so if u+v=0 then u is the same as v and distance is 0. 
    # set totals to 1 in this case to allow computation.
    
    d = ((pu - pv)**2/(x)).sum()
    d /= 2
    return d

def discrete_prob_dist(x, nbins=30):
    if np.any(np.isnan(x)):
        raise ValueError("Can't calculate NaN")

    prob, bins = np.histogram(x, bins=nbins)
    prob = prob / x.shape[0]
    bw = bins[1]-bins[0]

    rv = stats.rv_histogram((prob,bins))
    p = rv.pdf(x)*bw
    # x.max() is given 0 probability as not in last bin 
    # correct by getting pvalue of midpoint of its bin
    p[p==0] = rv.pdf(x.max()-bw/2)*bw 
    return p

def probability_matrix(X):
    P = np.apply_along_axis(discrete_prob_dist,0,X)
    return P

def pairwise_sim(X, metric, norm=False, kernel=False):
    assert isinstance(metric, str)

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    
    if metric in ['jensenshannon', 'chi_square']:
        P = probability_matrix(X)
        if metric=='chi_square':
            metric = chi_square
        D = dist.pdist(P, metric)
    else:
        D = dist.pdist(X, metric)

    # tidy up D 
    if metric in ['cosine','correlation']:
        # if ||u-ubar|| is 0. correlation is nan. 
        # same issue for cosine
        # Instead set correlation to 0. Distance is calculated as 1 - correlation
        # so we replace nan with 1
        # another problem if x*u is 0 i.e. they are perpindicular. 
        # Euclidean distance not max but cosine wise they are
        D = np.nan_to_num(D, nan=1.0)
    if metric=='jensenshannon':
        # scipy implementation of js can mess up if dist is 0.
        # if very close to 0 e.g. -1e-16 np.sqrt won't work
        # and will set as nan. replace with 0 in this case.
        D = np.nan_to_num(D, 0)
    
    # normalise/kernel
    if kernel:
        D = np.exp(-(D**2)/D.std())
    if norm:
        # want to have Distance in terms of standard deviations. 
        # this does stop us from easily identifying 0 i.e. identical items.
        # might be minimum might not. something to think about in future 
        # but for our purposes not an issue right now.
        # not sure about using nanmean rather than normal version
        # but nan shouldn't be there so this allows us to get normal numbers
        s = np.nanstd(D,dtype=np.float64)
        m = np.nanmean(D,dtype=np.float64)
        if np.any(np.isnan([m,s])):
            print(f'mean:{m} std: {s}')
        D = (D - m)/s
        
        # we don't want to normalise with diagonal values included
        # but we do want diagonal to be mapped to equivalent of 0 under the normalisation
        D = dist.squareform(D)
        id = (0 - m)/s
        np.fill_diagonal(D, id)

    if len(D.shape) == 1:
        D = dist.squareform(D)

    return D

def snf_affinity(X, metric='euclidean', K=20, mu=0.5):
    if isinstance(X, dict):
        X = [v for v in X.values()]
    if isinstance(X, np.ndarray):
        if len(X.shape)==3:
            X = [X[i,:,:] for i in range(X.shape[0])]
    S = snf.make_affinity(X, K=K, metric=metric, mu=mu)
    return S

def multi_modal_similarity(data, metric, norm=True, snf_aff=False, K=20, mu=0.5):
    Nm = len(data)
    N, d = data[0].shape
    S = np.empty((Nm, N, N))
    S.fill(np.nan)
    for i, X in enumerate(data):
        idx = utils.non_nan_indices(X)
        if snf_aff:
            D = pairwise_sim(X[idx,:], metric=metric, K=K, mu=mu)    
        else:
            D = pairwise_sim(X[idx,:], metric, norm=norm)
        j,k = np.meshgrid(idx,idx)
        S[i, j, k] = D
    return S