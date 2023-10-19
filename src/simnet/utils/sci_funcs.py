import warnings

import numpy as np
import pandas as pd
from sklearn.utils import Bunch


def check_rng(rng):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)
    return rng


def linspace(min_, max_, step):
    num = round((max_ - min_)/step) + 1
    return np.linspace(min_, max_, num=num)

def random_sample(n, max, seed=1871702):
    rng = np.random.default_rng(seed)
    if max < n:
        n = max
    sample = rng.choice(max, size=n, replace=False)
    return sample

def non_nan_indices(X, offset=0):
    """Find indices that do not have all nan values.
      Amount of nans needed to be classified as a nan index can be adjusted with offset.

    Args:
        X (np.ndarray): two dimensional (Nxd) array. nan counted per row.
        offset (int): amount of columns with values to still be considered nan. 
                        i.e. a row with (d - offset) values missing is nan

    Returns:
        np.ndarray: 1d array of indices in range [0,N-1]
    """
    assert len(X.shape)==2, f"X must be 2 dimensional, not {len(X.shape)}"
    N, d = X.shape
    indices = np.arange(N)
    idx = np.isnan(X).sum(axis=1) >= d - offset
    idx = indices[~idx]
    return idx


def nanmean(x, allnanvalue=np.nan, **npkwds):
    """Function to compute np.nanmean and replace empty slice with 
    user value. Defaults to np.nan i.e. np.nanmean([np.nan,np.nan]) = np.nan.

    Args:
        x (np.ndarray): array to apply np.nanmean to.
        allnanvalue (int, optional): Value in case of empty slice. Defaults to np.nan.
        **npkwds: keywords for np.nanmean function. e.g. axis=0 etc.
    Returns:
        _type_: nan mean of array or allnanvalue in case of empty slice in array
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        y=np.nanmean(x, **npkwds)
        y=np.nan_to_num(y, nan=allnanvalue)  
    return y

def dict2Bunch(datadict):
    return Bunch(**datadict)