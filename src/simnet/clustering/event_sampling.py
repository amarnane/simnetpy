import numpy as np
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform


"""
NOTE: event sampling does not currently work for CPM objective function. 
The function is not defined by Q and so the upper bound is not given by Q max.
Works perfectly for normal objective function!
"""


def plot_beta_curve(gamma_samples, beta_samples, yseq, bseq):
    # plt.style.use( 'seaborn-ticks')
    eps=0.01
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(bseq, yseq, label=r'$\gamma(\beta)$')
    ax.plot(beta_samples, gamma_samples, 'o', label='samples')

    lineargs = {'lw':1, 'ls':'dashed','color':'C1'}
    ax.vlines(beta_samples, 0, gamma_samples, **lineargs)
    ax.hlines(gamma_samples, 0, beta_samples, **lineargs)


    yupper = max(yseq.max(), gamma_samples.max())
    ax.set_yscale('log')
    ax.set_xlim(0,1+eps)
    ax.set_ylim(yseq.min()-eps, yupper)

    # locmin = mticker.LogLocator(base=10, subs=np.arange(0.1,yseq.max(),0.1),numticks=10)  
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_locator(tkr.LogLocator(base=10, subs='all'))
    ax.yaxis.set_minor_formatter(tkr.NullFormatter())
    # # ax.grid(True,which="both",ls="--",c='gray')  
    # ax.grid(True, which='both')
    ax.grid(False, which='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\gamma$')
    fig.show()
    return fig

def find_closest_value_gt(value, array):
    diff = array - value
    idx = np.where(diff > 0, diff, np.inf).argmin()
    return idx

def find_ymin(g, obj_func='modularity', step=0.1, max_significant_figures=5, verbose=False):
    assert len(g.clusters()) == 1, "g needs to be a connected graph (1 connected component)"
    args = {'objective_function':obj_func, 'n_iterations':-1}

    y=1 # starting resolution
    # step = 0.1 # 
    factor = 1 # starting factor
    nc = g.vcount()

    LOOPLIMIT=1000
    count = 0

    while nc > 1 and factor <= max_significant_figures and count < LOOPLIMIT:
        cc = g.community_leiden(resolution_parameter=y, **args)
        nc = len(cc.sizes())
        # print(f'{y:.3f} {nc}')
        if nc == 1:
            ymin = y
            if verbose:
                print(f'{ymin:.3f}')
            y = y + step # go back one step and search in next decimal place.
            step = step/10
            factor += 1
            nc = 2
        else:
            y = y - step
            if y <= 0:
                y = y + step # go back one step and search in next decimal place.
                step = step/10
                factor += 1
                nc = 2
        count += 1
    return ymin

def betaf(y, A, P):
    M = A - y*P
    
    # also called antiferromagnetic
    E_ = M[M < 0] # set of indices that have less than expected edges for this resolution
    
    B = np.abs(E_).sum()/np.abs(M).sum()
    return B   

def gammaf(B, A, P, events, bevents):
    # need to find set of indices that are less than expected 
    # set that are more than expected
    # for the gamma this beta would produce
    # indices only change at specific beta values
    # first find events where change and corresponding betas
    # then find closest beta larger than current.
    Q = A/P

    idx = find_closest_value_gt(B,bevents)
    E_ = Q < events[idx] # 
    Ep = ~E_

    qu = A[E_].sum() + B*(A[Ep].sum() - A[E_].sum())
    ql = (1-B)*P[E_].sum() + B*(P[Ep].sum())
    return qu/ql


def sample_events(Q, A, P, n_to_approx_beta=100):
    """
    Function to approximate beta event curve. Speeds computation for large networks with a large number of events.

    Args:
        Q (np.ndarray): A/P matrix pre-calculated and in squareform (i.e. just triu entries)
        A (np.ndarray): Adjacency matrix
        P (np.ndarray): expected adjacency matrix 
                (configuration model assumed - k_i*k_j/2m where k_i is degree of ith node and m is number of edges in network)
        n_to_approx_beta (int, optional): This is the number of samples used to appoximate the event curve. Defaults to 50.
                                Note: 50 other samples are used in the approximation although these are kept fixed 
                                so total is n_to_approx_beta+50 Defaults to 100.

    Returns:
        events, bevents: sequence of precalculated gamma and beta events.
    """
    # Find unique events
    events = np.unique(Q)[1:] # ignore 0 

    # Event curves are very skewed with large number of events occuring in highest 50%
    # This is also the area of least interest as most communities are single or isolated nodes.
    # we take look each percentile between 50 and 100 i.e. 51%, 52%, ...
    qq = np.linspace(0.5,1, num=51)
    upper_vals = np.array([np.quantile(events,q) for q in qq])

    # we only want to approximate the lower events with nsample points
    mid = np.median(events)
    events = events[events < mid]

    
    step = events.shape[0] // n_to_approx_beta
    # if there are less than nsample events step=0 which raises error.
    # use stepsize of 1 in this case. equivalent to n_to_approx_beta = events.shape[0]
    if not step:
        step = 1
    # sample an event after every step events (step dictated by n_to_approx_beta)
    events = events[::step]

    # calculate beta value for each event
    events = np.concatenate((events, upper_vals))
    bevents = np.array([betaf(e, A, P) for e in events])

    return events, bevents

def resolution_event_samples(g, n=15, plot=False, n_to_approx_beta=50, return_dict=False):
    """Function to identify a good sequence of resolutions for resolution parameter in leiden clustering.
    Samples are evenly spaced over the difference levels of hierarchy. From all nodes in single cluster to each node in individual cluster.
    Implementation of the method described in paper doi: 10.1038/s41598-018-21352-7

    Args:
        g (ig.Graph): igraph network to be clustered
        n (int, optional): length of sequence of resolutions to find. Defaults to 15.
        plot (bool, optional): Flag to plot beta-gamma event sample curve (as in the paper). Defaults to False.
        n_to_approx_beta (int, optional): In large networks, the number of events can be very large. Finding all beta events can take a long time. 
                                This is the number of samples used to appoximate the event curve. Defaults to 50.
                                Note: 50 other samples are used in the approximation although these are kept fixed so total is n_to_approx_beta+50.
                                
        return_dict (bool, optional): Flag to return beta samples as well as gamma samples. Defaults to False.

    Returns:
        _type_: _description_
    """

    A = g.get_adjacency_sparse()
    A = A.toarray()
    np.fill_diagonal(A,0) # don't want self loops

    D = A.sum(axis=0)


    P = np.outer(D,D)/(D.sum())
    np.fill_diagonal(P, 0)

    P = squareform(P)
    A = squareform(A)

    Q = A/P
    ymax = Q.max()
    ymin = find_ymin(g)

    Bmin = betaf(ymin, A, P)
    Bmax = betaf(ymax, A, P)

    events, bevents = sample_events(Q, A, P, n_to_approx_beta=n_to_approx_beta)

    beta_samples = np.linspace(Bmin,Bmax, num=n, endpoint=False)
    samples = np.array([gammaf(B, A, P, events, bevents) for B in beta_samples])

    if plot:
        yseq = np.logspace(np.log(ymin),np.log10(ymax), num=n_to_approx_beta)
        bseq = np.array([betaf(yy,A,P) for yy in yseq])

        fig = plot_beta_curve(samples, beta_samples, yseq, bseq)
    
    if return_dict:
        {'gamma_samples': samples, 'beta_samples': beta_samples}
    else:
        return samples

def find_max_mod_gamma(gamma_seq, g, obj_func='modularity', beta=0.01):
    args = {'objective_function':obj_func, 'n_iterations':-1, 'beta':beta}

    modularity = []
    for gamma in gamma_seq:
        clstr = g.community_leiden(resolution_parameter=gamma, **args)
        mod = clstr.modularity
        modularity.append(mod)

    idx = np.argmax(modularity)
    gamma_max = gamma_seq[idx]
    return gamma_max