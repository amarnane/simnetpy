import numpy as np
from sklearn.cluster import KMeans

# import graph_tool.all as gt
import warnings
warnings.simplefilter("always", ImportWarning)
def custom_import_warning(message):
    warnings.warn(message, ImportWarning)
    
try:
    import graph_tool.all as gt
except ImportError:
    gt='NOT INSTALLED'
    custom_import_warning("The graph-tool module is not installed. SBM clustering unavailable. Use\n\t conda install "+\
                "-c conda-forge graph-tool\nto install in a conda environment."+\
                " See https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions "+\
                "for help.")
#     warnings.warn("The graph-tool module is not installed. SBM clustering unavailable. Use\n\t conda install "+\
#                 "-c conda-forge graph-tool\nto install in a conda environment."+\
#                 " See https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions "+\
#                 "for help.", ImportWarning)
# # except ModuleNotFoundError:

from .event_sampling import resolution_event_samples
from .spectral import Spectral
from ..utils import linspace

# sbm clustering
def relabel(y):
    for i, c in enumerate(np.unique(y)):
        y[y ==c] = i
    return y

def sbm_clustering(g, deg_corr=False, wait=10, nbreaks=2, beta=np.inf, mcmc_niter=10):
    """Fit stochastic block model on g. 

    Args:
        g (gt.Graph or ig.Graph): graph to cluster
        deg_corr (bool, optional): flag to use degree corrected model. Defaults to False.
        wait (int, optional): number of steps required without record breaking event to end mcmc_equilibrate.
                                Defaults to 10.
        nbreaks (int, optional): number of times `wait` steps need to happen consecutively. Defaults to 2.
                                i.e. wait steps have to happen nbreaks times without a better state occuring.
        beta (float (or np.inf), optional): inverse temperature. controls types of proposal moves. beta=1 concentrates 
                    on more plausible moves, beta=np.inf performs completely random moves. Defaults to 1.
                    exact detail: epsilon in equation 14) https://arxiv.org/pdf/2003.07070.pdf
                                & epsilon in equation 3) https://journals.aps.org/pre/pdf/10.1103/PhysRevE.89.012804
        mcmc_niter (int, optional): number of gibbs sweeps to use in mcmc_merge_split. Defaults to 10.
                                Higher values give better proposal moves i.e. quality of each swap improves but 
                                time spent on each step in monte carlo should be minimised.
                                Discussion found in page 7 https://arxiv.org/pdf/2003.07070.pdf (parameter M used to estimate xhat)

    Returns:
        np.ndarray: cluster labels
    """
    if gt == "NOT INSTALLED":
        raise ImportError("The graph-tool module is not installed. SBM clustering unavailable. Use\n\t conda install "+\
                "-c conda-forge graph-tool\nto install in a conda environment."+\
                " See https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions "+\
                "for help.")
    
    if not isinstance(g, gt.Graph):
        g = g.to_graph_tool()

    state = gt.minimize_blockmodel_dl(g, state_args={'deg_corr':deg_corr})
    gt.mcmc_equilibrate(state, wait=wait, nbreaks=nbreaks, mcmc_args=dict(niter=mcmc_niter, beta=beta))

    y_pred = state.get_blocks().get_array() #state.b.a
    y_pred = relabel(y_pred)
    return y_pred

def sbm_clustering_entropy(g, deg_corr=False, wait=10, nbreaks=2, beta=np.inf, mcmc_niter=10):
    if gt == "NOT INSTALLED":
        raise ImportError("The graph-tool module is not installed. SBM clustering unavailable. Use\n\t conda install "+\
                "-c conda-forge graph-tool\nto install in a conda environment."+\
                " See https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions "+\
                "for help.")
    

    if not isinstance(g, gt.Graph):
        g = g.to_graph_tool()

    state = gt.minimize_blockmodel_dl(g, state_args={'deg_corr':deg_corr})
    gt.mcmc_equilibrate(state, wait=wait, nbreaks=nbreaks, mcmc_args=dict(niter=mcmc_niter, beta=beta))

    y_pred = state.get_blocks().get_array() #state.b.a
    y_pred = relabel(y_pred)
    entropy = state.entropy()
    return {'y_pred':y_pred, 'entropy': entropy}

# leiden clustering
def find_max_mod_gamma_clstr(gamma_seq, g, obj_func='modularity', beta=0.01):
    args = {'objective_function':obj_func, 'n_iterations':-1, 'beta':beta}

    modularity = []
    labels = []
    for gamma in gamma_seq:
        clstr = g.community_leiden(resolution_parameter=gamma, **args)
        mod = clstr.modularity
        modularity.append(mod)
        labels.append(clstr.membership)

    idx = np.argmax(modularity)
    gamma_max = gamma_seq[idx]
    y = labels[idx]
    return np.array(y), gamma_max

def leiden_single_component_clustering(g, obj_func='modularity', beta=0.01, nsamples=15):
    """cluster graph using leiden method. Uses event sampling to identify resolution parameter.
    resolution with maximum modularity is selected. Event sampling gives sequence of resolution 
    paramater that have more range of cluster numbers than a linear or logarithmic sequence. evenly covers
    one single cluster to all nodes are a separate cluster.  

    Args:
        g (ig.Graph): graph to cluster
        obj_func (str, optional): function for leiden method. Defaults to 'modularity'. Other option CPM
        beta (float, optional): randomness in leiden algorithm (only used in refinement step). Defaults to 0.01.
        nsamples (int, optional): number of samples to use to approximate event curve in event sampling. Defaults to 50.
                                    Higher more accurate but incredibly slow in large networks with many different events.

    Returns:
        np.ndarray: cluster labels
    """    
    gammasq = resolution_event_samples(g,n=nsamples)
    y_pred, gamma_max = find_max_mod_gamma_clstr(gammasq, g, obj_func, beta)
    return y_pred


# note with this approach we had to tackle problem connected components not being in order of size.
# say we have 3 components [0, 1, 2] if the large component was orignally idx 1 and had it two subclusters [0,1]
# then or method would have cluster labels as [0, 1, 2, 4] 0,1 corresponding to clusters in large component
# 2,4 corresponding to 0,2 in original cluster idxs. we refactor so our 4 clusters go from 0 to 3 [0,1,2,3]
def leiden_multi_component_clustering(g, cc=None, large_comp_cutoff=0.1, obj_func='modularity', beta=0.01, nsamples=15):
    # find connected components
    if cc is None:
        cc = g.connected_components()

    # create arrays for connect component membership & y_pred
    ccnp = np.array(cc.membership)
    y_pred = np.zeros(ccnp.shape[0])
    
    # find components larger than cutoff
    lcomps, idxs = g.large_components(large_comp_cutoff=large_comp_cutoff, return_idx=True)
    
    # apply leiden clsutering to each component
    # we keep track of current total clusters so each cluster has unique id
    nclstr = 0
    for gg, idx in zip(lcomps, idxs):
        # cluster component
        yi = leiden_single_component_clustering(gg, obj_func=obj_func, beta=beta, nsamples=nsamples)
        # assign unique id with nclstr
        y_pred[ccnp==idx] = yi + nclstr
        #update total number of clusters
        nclstr += len(np.unique(yi))

    # update idxs of smaller components
    disconnected_components = np.isin(ccnp, idxs, invert=True)
    y_pred[disconnected_components] = ccnp[disconnected_components] + nclstr

    # refactor cluster ids to run from 0 to K-1
    for i,k in enumerate(np.unique(y_pred)):
        y_pred[y_pred==k] = i
    return y_pred

def leiden_clustering(g, large_comp_cutoff=0.1, obj_func='modularity', beta=0.01, nsamples=15,  cc=None):
    # find connected components
    if cc is None:
        cc = g.connected_components()
    # check if fully connected
    if len(cc)==1:
        y_pred = leiden_single_component_clustering(g, obj_func=obj_func, beta=beta, nsamples=nsamples)
    else:
        y_pred = leiden_multi_component_clustering(g, cc=cc, large_comp_cutoff=large_comp_cutoff, 
                                                obj_func=obj_func, beta=beta, nsamples=nsamples)
    return y_pred


# spectral clustering
def spectral_clustering(g, laplacian='lrw', cmetric='cosine', max_clusters=50, min_clusters=2):
    """perform spectral clustering on graph on laplacian created from adjacency matrix. First Spectral decomp on laplacian. 
    Then uses eigengap to identify number of clusters K. Finally, clusters using K-means with user specified metric.
    
    cmet

    Args:
        g (ig.Graph or np.ndarray): graph to cluster (also accepts adjacency matrices)
        laplacian (str, optional): Select laplacian from random walk `lrw`, symmetric `lsym`,
                    unnormalised `l` or adjacency `a`. Defaults to 'lrw'.
        cmetric (str, optional): metric to use in Kmeans cluster step. Any scipy pdist string or callable 
                                accepted. Defaults to 'cosine'.
        max_clusters (int, optional): max number of clusters to accept. Defaults to 50.
        min_clusters (int, optional): min number of clusters to accept. Defaults to 2 (min=1 may not work).

    Returns:
        np.ndarray: cluster labels
    """
    if not isinstance(g, np.ndarray):
        A = g.get_adjacency_sparse().toarray()
    else:
        A = g

    spect = Spectral(laplacian_type=laplacian, custom_dist=cmetric, min_clusters= min_clusters, max_clusters=max_clusters)
    y_pred = spect.predict_from_adj(A)
    return y_pred


def baseline_kmeans_Kknown(X, k):
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(X)
    y_pred = kmeans.labels_
    return y_pred

# def baseline_kmeans(X, kmin=2, kmax=15, step=1, klist=[]):
def baseline_kmeans(X, kmin=2, kmax=15, step=1, klist=[]):
    if not klist:
        klist = linspace(kmin, kmax, step)

    # get reference inertia of all in same cluster
    inertia_o = np.square((X - X.mean(axis=0))).sum()
    # fit k-means for different K
    inertia = []
    ys = []
    for k in klist:
        k = int(k)
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(X)
        
        # criteria for selecting K - punishes larger number of clusters
        # every increase in number of clusters requires a 1/sqrt(N) improvement in interia.
        # note: this was chosen arbitrarily.  could also use alpha factor?
        scaled_inertia = kmeans.inertia_ / inertia_o + k/np.sqrt(X.shape[0]) # penalise number of clusters
        inertia.append(scaled_inertia)
        ys.append(kmeans.labels_)

    idx = np.argmin(inertia)
    y_pred= ys[idx]

    return y_pred

def baseline_spectral(X=None, S=None, metric='euclidean', norm=True, **spectkwds):
    spect = Spectral(**spectkwds)
    y_pred = spect.predict_from_aff(X=X, S=S, metric=metric, norm=norm)

    return y_pred




