import numpy as np
from sklearn.utils import Bunch

from .. import utils
from .distributions import equal_split, mixed_categorical_clusters, mixed_multi_numeric


def shuffle_within_groups(y, rng=None):
    rng = utils.check_rng(rng)

    idx = np.zeros(y.shape, dtype=np.int32)
    ids = np.arange(len(y))
    for k in np.unique(y):
        idxs = ids[y == k]
        shuffled = rng.permutation(idxs)
        idx[idxs] = shuffled
    return y[idx], idx


def merge_clusters(y, n, ns, rng=None):
    assert ns < n, "Error: in order to merge clusters ns must be less than n"

    rng = utils.check_rng(rng)

    # sample n cluster labels. Merge clusters based on new label.
    # list of cluster labels for merged clusters
    ys = np.arange(ns)
    # sample n-ns labels from the merged clusters
    merge = rng.choice(ys, size=n - ns)
    # we include ys to ensure each cluster included at least once.
    ym = np.concatenate((ys, merge))
    rng.shuffle(ym)  # shuffle to ensure guaranteed labels are random

    ymerge = np.zeros_like(y)
    for k, v in enumerate(ym):
        ymerge[y == k] = v

    return ymerge


def split_clusters(y, n, nb, shuffle=True, rng=None):

    rng = utils.check_rng(rng)

    assert nb > n, "Error: in order to split clusters nb must be greater than n"
    # Sample nm cluster labels. Split clusters based on old label
    # list of original labels
    yc = np.arange(n)

    # sample nb - n labels from old labels
    split = rng.choice(yc, size=nb - n)
    # we include yc to ensusure each label included at least once
    ysp = np.concatenate((yc, split))
    rng.shuffle(ysp)  # shuffle to ensure guaranteed labels are random

    _, splits = np.unique(ysp, return_counts=True)
    _, sizes = np.unique(y, return_counts=True)
    split_sizes = []
    for nsub, gsize in zip(splits, sizes):
        subgroups = equal_split(gsize, nsub)
        split_sizes.append(subgroups)
    split_sizes = np.concatenate(split_sizes)
    ysplit = np.concatenate(
        [val * np.ones(size) for val, size in zip(np.arange(nb), split_sizes)]
    )

    if shuffle:
        _, idx = shuffle_within_groups(y, rng=rng)
        ysplit = ysplit[idx]

    return ysplit


def fmerge(y, n, fixed=False, nlower=None, rng=None):
    # if rng is None:
    #     rng = np.random.default_rng()
    rng = utils.check_rng(rng)

    if nlower is None:
        nlower = n // 2
    if fixed:
        ns = nlower
    else:
        ns = rng.integers(nlower, n)
    yi = merge_clusters(y, n, ns, rng=rng)
    return yi


def fsplit(y, n, fixed=False, nupper=None, rng=None):
    rng = utils.check_rng(rng)

    if nupper is None:
        nupper = 2 * n
    if fixed:
        nl = nupper
    else:
        nl = rng.integers(n + 1, nupper + 1)
    yi = split_clusters(y, n, nl, rng=rng)
    return yi


def fnormal(y, n, fixed=False, rng=None):
    return y


def funinformative(y, n, fixed=False, rng=None):
    rng = utils.check_rng(rng)
    return rng.permutation(y)


def sort_array_by_y(X, y):
    idx_sorted = np.argsort(y)
    idx = np.argsort(idx_sorted)
    return X[idx, :]


def normalise_features(X):
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def load_mixed_cluster_data(sizes, distribution, **data_params):
    # if rng is None:
    #     rng = np.random.default_rng()

    nclusters = len(sizes)

    if distribution == "categorical":
        args = ["N", "d", "alpha", "beta", "nlevels", "rng"]
        kwargs = {k: v for k, v in data_params.items() if k in args}
        data = mixed_categorical_clusters(nclusters=nclusters, sizes=sizes, **kwargs)
    else:
        args = ["N", "d", "std", "lower", "upper", "scale_ul_with_d", "rng", "norm"]
        kwargs = {k: v for k, v in data_params.items() if k in args}
        data = mixed_multi_numeric(
            nclusters=nclusters, sizes=sizes, distype=distribution, **kwargs
        )
        data.X = normalise_features(data.X)
    return data.X


def multi_mod_data(y, dists, ctypes, fixed_merge_split=False, rng=None, **data_params):
    assert len(dists) == len(
        ctypes
    ), "Distriubtion indicators must equal split inforamtion"
    if rng is None:
        rng = np.random.default_rng(seed=data_params["seed"])
    else:
        rng = utils.check_rng(rng)

    data_params["rng"] = rng

    dist_map = {0: "guassian", 1: "studentt", 2: "categorical"}
    fy = {0: fnormal, 1: fmerge, 2: fsplit, 3: funinformative}
    _, sizes = np.unique(y, return_counts=True)
    n = sizes.shape[0]

    dataset = {"y": y, "Xi": [], "yi": []}

    for dist, ceffect in zip(dists, ctypes):

        # ni = #generate nclstrs in function
        yi = fy[ceffect](y, n, fixed=fixed_merge_split, rng=rng)

        _, sizes_i = np.unique(yi, return_counts=True)
        dist_i = dist_map[dist]
        Xi = load_mixed_cluster_data(sizes_i, dist_i, **data_params)  # load data

        # sizes_i is a sorted array. yi is random. we need to map Xi to yi's ordering
        Xi = sort_array_by_y(Xi, yi)

        dataset["Xi"].append(Xi)
        dataset["yi"].append(yi)

    dataset["m_prop"] = {"dists": dists, "ctypes": ctypes}
    return Bunch(**dataset)


###################### Example Settings ######################################
def multi_5_mod_settings():
    settings = {
        "easy": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "noisy": [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        "cat": [[2, 2, 2, 2, 2], [0, 0, 0, 0, 0]],
        "merged": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        "split": [[0, 0, 0, 0, 0], [2, 2, 2, 2, 2]],
        "mixed_normal": [[0, 0, 0, 0, 0], [1, 2, 1, 1, 2]],
        "mixed_noisy": [[1, 1, 1, 1, 1], [1, 2, 1, 1, 2]],
        "mixed_all": [[0, 0, 2, 1, 1], [1, 2, 1, 1, 2]],
        "half_noisy": [[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]],
        "half_merged": [[0, 0, 0, 0, 0], [0, 0, 1, 1, 2]],
        "1rand": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 3]],
        "noisy_1rand": [[1, 1, 1, 1, 1], [0, 0, 0, 0, 3]],
        "mixed_1rand": [[0, 0, 0, 0, 0], [1, 2, 1, 2, 3]],
        "mixed_noisy_1rand": [[1, 1, 1, 1, 1], [1, 2, 1, 2, 3]],
        "2rand": [[0, 0, 0, 0, 0], [0, 0, 0, 3, 3]],
        "3rand": [[0, 0, 0, 0, 0], [0, 0, 3, 3, 3]],
    }
    return settings


def multi_3_mod_settings():
    settings = {
        "easy": [[0, 0, 0], [0, 0, 0]],
        "noisy": [[1, 1, 1], [0, 0, 0]],
        "1rand": [[0, 0, 0], [0, 0, 3]],
        "merged": [[0, 0, 0], [1, 1, 1]],
        "mixed_normal": [[0, 0, 0], [1, 1, 2]],
        "mixed_noisy": [[1, 1, 1], [1, 1, 2]],
        "mixed_1rand": [[0, 0, 0], [1, 2, 3]],
        "noisy_1rand": [[1, 1, 1], [0, 0, 3]],
        "mixed_noisy_1rand": [[1, 1, 1], [1, 2, 3]],
        "split": [[0, 0, 0], [2, 2, 2]],
        "mixed_all": [[2, 0, 1], [1, 1, 2]],
        "single_noisy": [[0, 0, 1], [0, 0, 0]],
        "single_merged": [[0, 0, 0], [0, 0, 1]],
        "cat": [[2, 2, 2], [0, 0, 0]],
        "2rand": [[0, 0, 0], [0, 3, 3]],
    }
    return settings


def multi_mod_settings(nmod=3):
    if int(nmod) == 3:
        settings = multi_3_mod_settings()
    elif int(nmod) == 5:
        settings = multi_5_mod_settings()
    else:
        raise ValueError("nmod must be one of [3,5]")
    return settings


def create_multi_mod_data(
    data_params, dists=[0, 0, 0, 0, 0], ctypes=[0, 0, 0, 0, 0], rng=None
):
    N = data_params["N"]
    nclusters = data_params["nclusters"]

    sizes = equal_split(N, nclusters)
    y = np.concatenate([val * np.ones(size) for val, size in enumerate(sizes)])
    dataset = multi_mod_data(y, dists, ctypes, rng=rng, **data_params)

    return dataset
