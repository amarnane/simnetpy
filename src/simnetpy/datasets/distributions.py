import numpy as np
from scipy import stats
from sklearn.utils import Bunch


def uniform_sampler(d, std=1, rng=None):
    """sample random location in d dimensional box with
    max value of std and min value -std on any axis

    Args:
        d (int): number of dimensions
        std (float, optional): max abs value on any axis. Defaults to 1.
        rng (np.random.default_rng, optional): user seeded random number generator. Defaults to None.

    Returns:
        np.ndarray: random point in d dimensional space with max abs value of std on any dimension
    """
    if rng is None:
        rng = np.random.default_rng()

    a = [1, -1]

    direction = rng.choice(a, size=d)
    r = rng.uniform(0, std, size=d)
    return direction * r


def multivariate_guassian(N, center, std=1, rng=None):
    """Sample N point from a multivariate guassian with mean at center and
    Covariance of std*Identity (i.e. circular guassian). Dimensions of guassian inferred
    from user passed center.

    Args:
        N (int): number of points to sample
        center (np.ndarray): d-dimensional point
        std (float, optional): Standard deviation along any axis. Covariance is `std`*Identity Matrix. Defaults to 1.
        rng (np.random.default_rng, optional): user seeded random number generator. Defaults to None.

    Returns:
        np.ndarray: N samples from multi-dimensional guassian centered at `center`. (N x d) matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(std, np.ndarray):
        d = center.shape[0]
        COV = std * np.eye(d)
    else:
        COV = std

    X = rng.multivariate_normal(center, COV, size=N)
    return X


def multivariate_t(N, center, std=1, df=1, rng=None):
    """Sample N point from a multivariate guassian with mean at center and
    Covariance of std*Identity (i.e. circular guassian). Dimensions of guassian inferred
    from user passed center.

    Args:
        N (int): number of points to sample
        center (np.ndarray): d-dimensional point
        std (float, optional): Standard deviation along any axis. Covariance is `std`*Identity Matrix. Defaults to 1.
                                Also accepts numpy covariance matrix
        df (float, optional): Degrees of freedom of the distribution. Defaults to 1. If np.inf results are multivariate normal.
        rng (np.random.default_rng, optional): user seeded random number generator. Defaults to None.

    Returns:
        np.ndarray: N samples from multi-dimensional guassian centered at `center`. (N x d) matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(std, np.ndarray):
        d = center.shape[0]
        COV = std * np.eye(d)
    else:
        COV = std

    # X = rng.multivariate_normal(center, COV, size=N)
    # dist = stats.multivariate_t(center, shape=COV, df=df, seed=rng)
    # X = dist.rvs(size=size)
    X = stats.multivariate_t.rvs(loc=center, shape=COV, df=df, size=N, random_state=rng)

    return X


def cluster_centers(n, d, lower=1, higher=2, rng=None, init=None):
    """Sample n point in d-dimensional space. Points will be between
    (lower, ~2*higher) distance from all other points. Initial center will be (0,0, ..,0)
    unless otherwise specified. Sampling done sequentially with next center proposed,
    rejected if too close to all others and resampled until accepted.
    If sampling large number of points & time taken is large increase size of higher.

    Note: 2*higher is not actual distance upper bound. higher controls size of box around previous center
    that we sample a proposal point. Each center sampled from box with sides of size 2*higher.
    Args:
        n (int): number of points to generate
        d (int): number of dimensions
        lower (float, optional): Lower bound of distances to accept. All points will be
                            at least lower away from each other. Defaults to 1.
        higher (float, optional): Size of box around previous center to sample from. Defaults to 2.
        rng (np.random.default_rng, optional): user seeded random number generator. Defaults to None.
        init (np.ndarray, optional): Location of first sample. Defaults to None. Note points are shuffled
        but if none at least one will be the origin (0,0,...,0).

    Returns:
        list: n randomly sampled points in d-dimensional space all at least lower away from each other.
    """
    if rng is None:
        rng = np.random.default_rng()

    if init is None:
        init = np.zeros(d)
    centers = []
    x = init + uniform_sampler(d, std=higher, rng=rng)
    centers.append(x)

    while len(centers) < n:
        # sample point in random direction
        x = uniform_sampler(d, std=higher, rng=rng)

        # find random center and move away in direction x
        i = rng.integers(0, len(centers))
        x = centers[i] + x

        carray = np.array(centers)
        dist = np.linalg.norm(carray - x, axis=1, ord=2)

        if np.all(dist > lower):
            centers.append(x)
    rng.shuffle(centers)  # shuffle so first cluster is not necessarily close to init
    return centers


def mixed_multi_numeric(
    nclusters,
    d,
    N=100,
    std=1,
    lower=2,
    upper=5,
    sizes=None,
    distype=None,
    scale_ul_with_d=False,
    df=2,
    rng=None,
):
    if isinstance(sizes, str):
        assert sizes.lower() in [
            "equal",
            "random",
            "roughly_equal",
        ], "if specifying method sizes must be one of [equal, random, roughly_equal]"
        sizes = split_data_into_clusters(N, nclusters, method=sizes)
    elif sizes is None:
        sizes = split_data_into_clusters(N, nclusters, method="equal")

    if scale_ul_with_d:
        lower = lower / np.sqrt(d)
        upper = upper / np.sqrt(d)

    centers = cluster_centers(n=nclusters, d=d, lower=lower, higher=upper, rng=rng)

    X = []
    y = []
    for i, (cc, m) in enumerate(zip(centers, sizes)):
        if distype == "studentt":
            x = multivariate_t(m, cc, std=std, df=df, rng=rng)
        else:
            x = multivariate_guassian(m, cc, std=std, rng=rng)
        X.append(x)

        labels = [i] * m
        y += labels

    return Bunch(y=np.array(y), X=np.vstack(X))


# def polygenerator(degree, upperb, rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     coef = rng.uniform(low=-upperb, high=upperb, size=degree + 1)
#     polyt = np.poly1d(coef)
#     return polyt


# def polytransform(X, degree, upperb, rng=None, norm=False):
#     polyt = polygenerator(degree, upperb, rng)

#     X = polyt(X)
#     if norm:
#         X = (X - X.mean(axis=0)) / X.std(axis=0)
#     return X


# def multi_modal_guassian(
#     nmodality=3, nclusters=3, N=50, d=10, std=1, sizes=None, rng=None, normalise=True
# ):
#     if rng is None:
#         rng = np.random.default_rng()

#     if isinstance(sizes, str):
#         assert sizes.lower() in [
#             "equal",
#             "random",
#             "roughly_equal",
#         ], "if specifying method sizes must be one of [equal, random, roughly_equal]"
#         sizes = split_data_into_clusters(N, nclusters, method=sizes)
#     elif sizes is None:
#         sizes = split_data_into_clusters(N, nclusters, method="equal")

#     # std = 1
#     lower = 3 / np.sqrt(
#         d
#     )  # this value found to work nicely between 2-10 dimensions. completely arbitrary.
#     upper = 7 / np.sqrt(d)  # similar to above.

#     data = {}

#     for i in range(nmodality):
#         dataset = mixed_multi_guassian(
#             nclusters, d, N=N, std=std, lower=lower, upper=upper, sizes=sizes, rng=rng
#         )
#         X = dataset.X
#         y = dataset.y

#         if normalise:
#             X = (X - X.mean(axis=0)) / X.std(axis=0)

#         data[f"X{i}"] = X
#     # data['y'] = y

#     return Bunch(data=data, y=y)


# def multi_modal_data(
#     nmodality=3,
#     nclusters=3,
#     N=300,
#     d=10,
#     std=1,
#     rng=None,
#     nonlinear=True,
#     max_degree=8,
#     coef_max_abs=2,
#     norm=True,
# ):

#     dataset = multi_modal_guassian(nmodality, nclusters, N, d, std, rng)

#     if nonlinear:
#         y = dataset.y
#         data = {}
#         for k, X in dataset.data.items():
#             degree = rng.integers(2, max_degree + 1)
#             # print(degree)
#             Xt = polytransform(X, degree, coef_max_abs, rng, norm=norm)
#             data[k] = Xt

#     return Bunch(data=data, y=y)


############################# Categorial ###############################################


def random_discrete_pmf(
    nlevels=5,
    min_largest_cat=0,
    max_any_cat=None,
    min_each_cat=0,
    shuffle=True,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    assert (
        min_each_cat * nlevels < 1
    ), f"error guaranteeing each cat has minimum {min_each_cat} prob mass results in sum(prob mass) > 1"

    total = 1 - min_each_cat * nlevels  # total probability mass to distribute

    if max_any_cat is None:
        max_any_cat = total

    assert (
        max_any_cat <= total and max_any_cat >= 1 / nlevels
    ), f"max_any_cat must be float between 1/nlevels and {total}."
    assert (
        min_largest_cat <= total and min_largest_cat >= 0
    ), f"min_largest_cat must be float between 0 and {total}"
    assert (
        max_any_cat >= min_largest_cat
    ), "allowable upper bound for any category must be larger than minimum value for largest"

    p = np.zeros(nlevels)
    # total = 1 # total probability mass to distribute

    upper = max_any_cat  # max in
    for i in range(nlevels - 1):
        if not i:
            lower = (
                min_largest_cat  # if we want to skew the distribution we set a value
            )
            # above 0.5 to that at prob of first element is > 0.5
        else:
            lower = 0  # for all other elements we want to equally distriute the remaining prob mass

        if total < max_any_cat:  # when remaining probability mass less than upper bound
            upper = total

        # when limiting maximum of any category with max_any_cat
        # the lower bound needs to be adjusted in case a sequence of too small values are sampled.
        # otherwise final value p_{n-1} will be greater than max_any_cat
        # lower is the minimum value possible assuming the remaining categories are assigned the max_any_cat value.
        lower = total - (
            max_any_cat * (nlevels - i - 1)
        )  # i is number of categories sampled. nlevels -i -1 = number of categories remaining -1
        lower = max(lower, 0)

        x = rng.uniform(low=lower, high=upper, size=1)
        total = total - x
        p[i] = x

    # xn = 1 - sum(p)
    p[-1] = total

    if shuffle:
        rng.shuffle(p)

    if min_each_cat:
        p += min_each_cat

    return p


def roughly_equal_split(N, nclusters, margin=None, rng=None):
    # note could also change starting average cluster size
    # if [N*frac]*nclusters Then max_margin is max_margin=1/(ncluster-1) - frac
    # as e.g. if frac=1/(N+1) then immediate increase in possible margin sizes that can be used
    # note: assert frac < 1/ncluster otherwise total is too large
    if rng is None:
        rng = np.random.default_rng()

    max_margin = 1 / (nclusters - 1) - 1 / nclusters
    if margin is None:
        margin = max_margin
    assert margin <= max_margin, f"margin must be less than max possible: {max_margin}"

    shift = rng.uniform(-margin, margin, nclusters - 1)
    if np.all(
        shift == max_margin
    ):  # in very unlikely event that all are max_margin resample
        shift = rng.uniform(-margin, margin, nclusters - 1)

    shift = np.round(shift * N)

    equal = [N // nclusters] * nclusters
    sizes = np.zeros(nclusters, dtype="int")
    for i, size in enumerate(equal[:-1]):
        sizes[i] = size + shift[i]
    sizes[-1] = N - sizes[:-1].sum()

    return sizes


def random_split(
    N,
    nclusters,
    max_any_cat=0.6,
    min_each_cat=0,
    min_largest_cat=0,
    shuffle=True,
    rng=None,
):

    sizes = random_discrete_pmf(
        nlevels=nclusters,
        max_any_cat=max_any_cat,
        min_largest_cat=min_largest_cat,
        min_each_cat=min_each_cat,
        shuffle=shuffle,
        rng=rng,
    )
    sizes = np.round(N * sizes).astype("int")
    # resample if empty
    if np.any(sizes == 0):
        sizes = random_split(N, nclusters, max_any_cat=max_any_cat)

    # check for rounding errors
    rounderror = N - sizes.sum()

    for i in range(np.abs(rounderror)):
        if rounderror > 0:
            sizes[i] += 1
        elif rounderror < 0:
            sizes[i] -= 1
    return sizes


def equal_split(N, nclusters):
    sizes = [N // nclusters] * nclusters  # split equally
    rounderror = N - sum(sizes)
    for i in range(rounderror):
        sizes[i] += 1
    return np.array(sizes)


def split_data_into_clusters(
    N, nclusters, method="equal", random_kwds={}, requal_kwds={}
):
    assert method in [
        "equal",
        "random",
        "roughly_equal",
    ], "Must be one of equal, random, roughly equal"

    if method == "equal":
        sizes = equal_split(N, nclusters)
    elif method == "random":
        sizes = random_split(N, nclusters, **random_kwds)
    else:
        sizes = roughly_equal_split(N, nclusters, **requal_kwds)
    return np.array(sizes)


def single_categorical_feature(
    N, nlevels, min_largest_cat=0.8, max_any_cat=1, shuffle=True, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    p = random_discrete_pmf(
        nlevels,
        min_largest_cat=min_largest_cat,
        max_any_cat=max_any_cat,
        shuffle=shuffle,
        rng=rng,
    )
    x = rng.choice(nlevels, size=N, p=p)
    return x


def mixed_categorical_cluster_feature(
    sizes, nlevels, min_largest_cat=0.8, max_any_cat=1, shuffle=True, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    # nlevels = 5
    X = []
    for size in sizes:
        x = single_categorical_feature(
            size,
            nlevels=nlevels,
            min_largest_cat=min_largest_cat,
            max_any_cat=max_any_cat,
            shuffle=shuffle,
            rng=rng,
        )
        # p = random_discrete_pmf(nlevels, min_largest_cat=min_largest_cat,
        #                         max_any_cat=max_any_cat, shuffle=shuffle, rng=rng)
        # # print(p.argmax(), p.max())
        # x = rng.choice(nlevels, size=size, p=p)
        X.append(x)
    return np.hstack(X)


def mixed_categorical_clusters(
    N,
    d,
    nclusters=3,
    sizes="equal",
    alpha=5,
    beta=1,
    nlevels=5,
    return_skew_factors=True,
    rng=None,
):
    """alpha and beta control shape of skew factor distribution
    higher max(abs(a,b))>1 -> less flat more peaked distribution
    a<b -> skewed below 0.5
    a>b -> skewed above 0.5
    so 5,1 means on average skew passed to ordinal feature generator will be centered on a/a+b~0.83
    and 1,5 means average skew will be ~0.16
    lower average skew means noisier data and harder clustering problem

    """

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(sizes, str):
        assert sizes.lower() in [
            "equal",
            "random",
            "roughly_equal",
        ], "if specifying method sizes must be one of [equal, random, roughly_equal]"
        sizes = split_data_into_clusters(N, nclusters, method=sizes)
    elif sizes is None:
        sizes = split_data_into_clusters(N, nclusters, method="equal")

    assert sizes.shape[0] == nclusters, "nclusters and sizes must match"
    assert sizes.sum() == N, f"sizes must add up to N {sizes.sum()} != {N}"

    # N = sizes.sum()

    # generate skew distribution & sample
    rv = stats.beta(a=alpha, b=beta)
    skew_factors = rv.rvs(size=d, random_state=rng)

    # generate features
    X = np.zeros((N, d))
    for i in range(d):
        si = skew_factors[i]
        if si >= 0.5 and si <= 1:
            skew = (
                2 * ((nlevels - 1) / nlevels) * si + (2 - nlevels) / nlevels
            )  # map skew factor from [0.5, 1] to [1/nlevels, 1] i.e. si=0.5, skew=1/nlevels
            X[:, i] = mixed_categorical_cluster_feature(
                sizes, nlevels=nlevels, min_largest_cat=skew, rng=rng
            )
        elif si >= 0 and si < 0.5:
            skew = (
                2 * ((1 - nlevels) / nlevels) * si + 1
            )  # map skew factor from [0.5, 0] to [1/nlevels, 1] i.e si=0 -> skew = 1.0
            X[:, i] = single_categorical_feature(
                N, nlevels=nlevels, min_largest_cat=skew, rng=rng
            )
        else:
            raise ValueError("Error beta distribution ill defined")

    # generate labels
    y = np.zeros(N)
    total = 0
    for i, size in enumerate(sizes):
        y[total : total + size] = i
        total += size

    dataset = Bunch(y=y, X=X)
    if return_skew_factors:
        dataset["skew_factors"] = skew_factors

    return dataset
