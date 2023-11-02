import math
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import seaborn as sns



# from ..graph import Igraph


def log_bin_means(x, y, bins):
    """Function to calculate the mean y value in each bin. 
    Finds y values for which corresponding x values are within the bin (endpoint not included).
    Mean is taken over all integers falling within the bin. 
    Mean value and bin midpoints are returned.
    e.g. x: [2,4, 8], y: [2, 1, 1], bin = [2, 6, 12]
    gives [4, 9], [(2+1)/4, 1/6]

    Args:
        x (np.array): sequence of node degrees 
        y (np.array): corresponding counts of each degree
        bins (np.array): sequence of bins to get average count in.

    Returns:
        (np.array, np.array): Returns (bin midpoints, mean value in bin)
    """
    yhat = []
    xmid = []
    for start, end in zip(bins[:-1], bins[1:]):
        midpoint = start + (end - start)/2
        start = round(start)
        end = round(end)
        counts = y[(x>=start) & (x < end)]
        n = end - start
        mean = counts.sum()/n
        yhat.append(mean)
        xmid.append(midpoint)
        
    return np.array(xmid), np.array(yhat)

def find_cutoff(step):
    """Find lowest integer where log(k+1) - log(k) < step

    Args:
        step (float): step taken in logspace

    Returns:
        int: cutoff below which integers increment more than *step* in logspace
        e.g. 5 cutoff for 0.1 as log(5) - log(4) = 0.097 but log(4) - log(3) > 0.1
    """
    seq = np.arange(1,50)
    d = np.diff(np.log10(seq))

    cutoff = seq[1:][d<step].min()
    return cutoff

def get_log_bins(x, y, step=0.1, verbose=False):
    """Function to create log bins for a degree sequence

    Args:
        x (np.array): sequence of node degrees 
        y (np.array): corresponding counts of each degree
        step (float, optional): stepsize in logspace. Lower increase resolution. Min step supported 0.01 Defaults to 0.1.
        verbose (bool, optional): Flag to print resolution and cutoff. Defaults to False.

    Returns:
        _type_: _description_
    """
    max_x = np.log10(x.max())
    min_x = np.log10(x[x>0].min())
    
    # Find next point to the right of max_x if incrementing in steps
    #  (e.g. 2.8 for max 2.73 when incrementing by 0.1)
    endpoint = round(math.ceil(max_x/step)*step,3)
    # find number of points for logspace to increase in steps of 0.1
    numpoints = math.ceil((endpoint-min_x)/step)

    # get bins
    bins = np.logspace(min_x, endpoint, num=numpoints)
    # find cutoff where an increment of 1 in linspace is less than step in logspace
    cutoff = find_cutoff(step)

    if verbose:
        print(f"Resolution: {step:.3f} Cutoff: {cutoff}")

    # only keep bins above cutoff
    bins = bins[bins>cutoff]

    # get mean value for each bin where
    # mean is sum of degree values / number of integers between the bin end points
    xm, ym = log_bin_means(x, y, bins)
    # combine degree & count values with bin midpoints and average count
    xlog = np.concatenate((x[x<=cutoff], xm))
    ylog = np.concatenate((y[x<=cutoff], ym))
    return xlog, ylog


def get_log_ticks(max_val, start=0):
    """helper function to display log ticks on degree plot"""

    if max_val < 1:
        NEG=True
        exp = math.floor(np.log10(max_val))
        logvals = np.linspace(start=exp, stop=0, num=np.abs(exp)+1)

    else:
        NEG=False
        # tmp = max_val
        # max_val = start
        # start
        exp = math.floor(np.log10(max_val))
        logvals = np.linspace(start=start, stop=exp, num=exp+1)

    # midvals are the points we place a tick. 
    # 1,2 correspond to 1,2 or 10, 20 or 100, 200 i.e. 1*10**x, 2*10**x, ...
    midvals = np.log10([1,2,3,4,5,6,7,8,9])
    ticks = np.tile(logvals,(midvals.shape[0],1))
    for i in range(midvals.shape[0]):
        if NEG:
            ticks[i,:] = ticks[i,:] - midvals[i]
        else:
            ticks[i,:] = ticks[i,:] + midvals[i]
    ticks = ticks.T.flatten()
    return ticks

def plot_degree_dist(degree_dist, logbinsize=0.1, LOG_ONLY=False):
    """
    Plot degree distribution as histogram and log-log scatter plot.
    Linear and log bin sequence shown for log-log plot
    
    Args:
        degree_dist (numpy array): sequence of node degrees.
        logbinsize (float, optional): stepsize in logspace. Lower increase resolution. 
                                Min step supported 0.01 Defaults to 0.1.
    """
    deg, cnts = np.unique(degree_dist, return_counts=True)
    
    if LOG_ONLY:
        fig, ax = plt.subplots(1,1, figsize=(8,6))

        sns.scatterplot(x=np.log10(deg), y=np.log10(cnts), ax=ax, alpha=0.7)
        ax.set_ylabel('Count', labelpad=15, fontsize=15)
        ax.set_xlabel('Degree', labelpad=15, fontsize=15)
        
        ## log binned dist
        deg, cnts = deg[1:], cnts[1:] # skip degree 0 as log(0) = inf
        
        x, y =  get_log_bins(deg, cnts, step=logbinsize)
        sns.scatterplot(x=np.log10(x), y=np.log10(y), ax=ax, alpha=0.7)
        # ax.plot(np.log10(x), np.log10(y), 'ro')

        # get log ticks to put on axis
        xticks = get_log_ticks(deg.max())
        yticks = get_log_ticks(cnts.max())
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        # display ticks as 10 to the power rather than 10, 100 etc
        formatter = lambda x, pos: f'{10 ** x:g}' if not x % 1 else ''
        ax.get_xaxis().set_major_formatter(formatter)
        ax.get_yaxis().set_major_formatter(formatter)

        # lims = np.log10(np.array([0.005, 25.]))
        # ax.set_xlim(lims)
        # ax.set_ylim(lims)
        # include degree distribution statistics
        # fig.text(0.8, 0.29, degstring, size=12)
        fig.suptitle('Degree Distribution', size=20)
        fig.show()
    else:
        # Plot degree distribution as lin-lin and log-log
        fig, ax = plt.subplots(2,figsize=(10, 8))

        # Fig 1 Linear distribution
        i=0
        sns.histplot(x=degree_dist, kde=True, binwidth=2, stat='probability',ax=ax[i])
        ax[i].set_ylabel('Probability', labelpad=15, fontsize=15)
        # include graph information
        # fig.text(0.8, 0.73, graphstring, size=12)
        
        # Fig 2 Log-Log distribution
        i = 1
        deg, cnts = deg[1:], cnts[1:] # skip degree 0 as log(0) = inf
        sns.scatterplot(x=np.log10(deg), y=np.log10(cnts), ax=ax[i], alpha=0.7)
        ax[i].set_ylabel('Count', labelpad=15, fontsize=15)
        ax[i].set_xlabel('Degree', labelpad=15, fontsize=15)
        
        ## log binned dist
        x, y =  get_log_bins(deg, cnts, step=logbinsize)
        sns.scatterplot(x=np.log10(x), y=np.log10(y), ax=ax[i], alpha=0.7)
        # ax[i].plot(np.log10(x), np.log10(y), 'ro')

        # get log ticks to put on axis
        xticks = get_log_ticks(deg.max())
        yticks = get_log_ticks(cnts.max())
        ax[i].set_xticks(xticks)
        ax[i].set_yticks(yticks)
        # display ticks as 10 to the power rather than 10, 100 etc
        formatter = lambda x, pos: f'{10 ** x:g}' if not x % 1 else ''
        ax[i].get_xaxis().set_major_formatter(formatter)
        ax[i].get_yaxis().set_major_formatter(formatter)

        # lims = np.log10(np.array([0.005, 25.]))
        # ax.set_xlim(lims)
        # ax.set_ylim(lims)
        # include degree distribution statistics
        # fig.text(0.8, 0.29, degstring, size=12)
        fig.suptitle('Degree Distribution', size=20)
        fig.show()


# plot clusters 
    
DEFAULT_VS = {
    'vertex_size' : 1,
    'edge_color' : "#b5b3b3",
    'bbox': (600, 600),
    'margin' : 20,
    'edge_curved': 0,
}

def color_nodes(ax, g, y, alpha=0.5):
    """Color node patches based on cluster labels
    (uses current mpl style colors)
    applies alpha to nodes and edges
    # passing color to igraph does not work with matplotlib backend

    Args:
        ax (plt.axis): axis to color vertex patches on
        g (_type_): Igraph graph instance
        y (list or np.ndarray): cluster labels
        alpha (float, optional): alpha setting for nodes and edges. Defaults to 0.5.
    """
    m = g.vcount() + g.ecount() # igraph plots first vertices then edges
    cc = ax.get_children()

    for i, c in enumerate(cc[:m]):
        c.set_alpha(alpha)
        # if a circle i.e. a vertex set color according to cluster membership
        if isinstance(c, ptc.Circle):
            c.set_edgecolor(f'C{y[i]}')
            c.set_facecolor(f'C{y[i]}')
    

def plot_graph_colored_nodes(g, y=None, ax=None, vs=None, alpha=0.5):
    """Plot network and cluster nodes

    Args:
        g (_type_): network to plot
        y (_type_, optional): cluster labels. Defaults to one single cluster.
        ax (_type_, optional): axis to plot on if given. Defaults to None.
        vs (_type_, optional): visual style dict for igraph. Defaults to DEFAULT_VS if nothing passed.
        alpha (float, optional): level of transparancey for nodes and edges. Defaults to 0.5.

    Returns:
        fig, ax: if ax passed fig is None.
    """
    if vs is None:
        vs = DEFAULT_VS

    if y is None:
        y = np.zeros(g.vcount(), dtype=np.int8)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    
    # if isinstance(g, Igraph):
    #     g = g.plottable()

    try:
        g = g.plottable()
    except AttributeError:
        g = g
    
    ig.plot(g, target=ax, **vs)
    color_nodes(ax, g, y, alpha)

    return fig, ax