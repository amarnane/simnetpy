import snf
import numpy as np

from .. import utils
import scipy.spatial.distance as dist

def make_positive(S):
    S = S - np.nanmin(S)
    return S

def standardise(S):
    np.fill_diagonal(S, 0)
    S = dist.squareform(S)
    s = np.nanstd(S,dtype=np.float64)
    m = np.nanmean(S,dtype=np.float64)
    S = (S - m)/s
    # we don't want to normalise with diagonal values included
    # but we do want diagonal to be mapped to equivalent of 0 under the normalisation
    S = dist.squareform(S)
    diag = (0 - m)/s
    np.fill_diagonal(S, diag)
    return S

def non_nan_standardise(S):
    idx = utils.non_nan_indices(S, offset=1)
    j,k = np.meshgrid(idx,idx)
    S[j,k] = standardise(S[j,k])
    return S

def extreme_mean(S):
    S1 = S.copy()
    if len(S.shape)==3:
        for i in range(S.shape[0]):
            S1[i,:,:] = non_nan_standardise(S1[i,:,:])
    else:
        S1 = non_nan_standardise(S1)

    mask = np.abs(S1) < 1
    S1[mask] = np.nan
    D = utils.nanmean(S1, allnanvalue=np.nanmax(S1), axis=0) # S1 is a distance so want to set to max if all nan
    return D

def nanmean(S):
    D = utils.nanmean(S, allnanvalue=np.nanmax(S), axis=0)
    return D

def mean_nan_max(S):
    S1 = S.copy()
    # replace nans with max distance. If i&j and have nan values between them we consider them less alike
    S1 = np.nan_to_num(S1, nan=np.nanmax(S1)) 
    # S1[np.isnan(S1)] = S1.max()
    D = np.mean(S1, axis=0)
    return D


def snf_fuse(S, K=20, iterations=20, alpha=1.0):
    S = S.copy() # don't change original array

    S = make_positive(-S) # convert from distance to similarity 
    np.nan_to_num(S, 0) # set nan values as 0.
    if isinstance(S, np.ndarray):
        S = [S[i,:,:] for i in range(S.shape[0])]
    # normalise to 0,1
    with np.errstate(divide='ignore',invalid='ignore'):
        D = -snf.snf(S, K=K, t=iterations, alpha=alpha) # SNF is similarity array. Our KNN constructors expect dissimilarity so set -D_snf
    return D

def rel_sim_nemo(D, K=20):
    D = D.copy() # don't change original array
    # We use dissimilarity but Rel Sim relies on non neighbour
    # entries being 0. So +ve values need to be indication of similarity 
    # convert to +ve similarity matrix with 0 being disimilar and max being most similar
    D = make_positive(-D)

    A = np.zeros(D.shape)    

    np.fill_diagonal(D, 0) # don't want self edges in rel sim calc.

    # Find KNN network
    nn = np.argsort(D, axis=1)[:,:K]

    non_nan_idx = utils.non_nan_indices(D, offset=1) # find all nan rows excluding diagonal
    # for i in range(nn.shape[0]):
    for i in non_nan_idx:
        A[i,nn[i]] = D[i, nn[i]]
    np.nan_to_num(A, 0)

    # Calc Relative Similarity
    Ai = A.sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        RS = A/Ai + A/Ai[:,np.newaxis]
    # np.nan_to_num(RS, 0)

    np.fill_diagonal(RS, np.nanmax(RS))
    RS = -RS # reconvert to dissimilarity

    return RS 

def avg_rel_sim_nemo(Smod, K=20):
    # Find relative sim of each modality
    S = np.zeros(Smod.shape)
    
    # S is a distance not a similarity
    # normalise to 0,1 in rel sim
    # Smod = make_postive(Smod)
    for i in range(S.shape[0]):
        S[i,:,:] = rel_sim_nemo(Smod[i,:,:], K=K)
    # Take nanmean (unlike NEMO we don't require at least one shared )
    S = nanmean(S)
    return S