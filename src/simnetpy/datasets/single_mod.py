import numpy as np

from .distributions import equal_split, random_split


################### CLUSTER PROBLEMS ###################
def large_cluster(
    N=2500, nclusters=10, min_largest_cat=0.5, min_each_cat=0.01, seed=2, verbose=False
):
    rng = np.random.default_rng(seed=seed)
    sizes = random_split(
        N=N,
        nclusters=nclusters,
        min_largest_cat=min_largest_cat,
        min_each_cat=min_each_cat,
        rng=rng,
    )
    if verbose:
        nbig = (sizes > 2 * (N / nclusters)).sum()
        nsmall = (sizes < N / (2 * nclusters)).sum()
        print(f"clusters > {2*(N/nclusters)}: {nbig}")
        print(f"clusters < {N/(2*nclusters)}: {nsmall}")

    return sizes


def mixed_large_small_clusters(
    N=2500, nclusters=10, min_each_cat=0.01, max_any_cat=0.5, seed=5, verbose=False
):
    rng = np.random.default_rng(seed=seed)
    sizes1 = random_split(
        N=N,
        nclusters=nclusters,
        max_any_cat=max_any_cat,
        min_each_cat=min_each_cat,
        rng=rng,
    )
    sizes2 = random_split(
        N=N,
        nclusters=nclusters,
        max_any_cat=max_any_cat,
        min_each_cat=min_each_cat,
        rng=rng,
    )
    sizes = (sizes1 + sizes2) // 2
    if verbose:
        nbig = (sizes > 2 * (N / nclusters)).sum()
        nsmall = (sizes < N / (2 * nclusters)).sum()
        print(f"clusters > {2*(N/nclusters):.2f}: {nbig}")
        print(f"clusters < {N/(2*nclusters):.2f}: {nsmall}")

    rounderror = N - sizes.sum()
    for i in range(rounderror):
        sizes[i] += 1
    return sizes


def equal_clusters(N=2500, nclusters=10):
    sizes = equal_split(N, nclusters)
    return sizes


def single_mod_cluster_problems(N):
    clust_problems = {}
    for nclusters in [3, 10, 30]:
        sizes = equal_clusters(N, nclusters)
        clust_problems[f"equal_{nclusters}"] = sizes

    nclusters = 10
    sizes = large_cluster(N, nclusters)
    clust_problems["single_large"] = sizes

    sizes = mixed_large_small_clusters(N, nclusters)
    clust_problems["mixed_sizes"] = sizes

    return clust_problems
