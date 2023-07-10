import numpy as np
from collections import Counter
from sklearn import metrics

from ..utils import nanmean

def cluster_accuracy(y_true, y_pred):
    vm = metrics.v_measure_score(y_true, y_pred)
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    # acc = metrics.accuracy_score(y_true, y_pred)
    # balacc = metrics.balanced_accuracy_score(y_true, y_pred)
    # f1 = metrics.f1_score(y_true, y_pred, average=None)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    homo = metrics.homogeneity_score(y_true, y_pred)
    complt = metrics.completeness_score(y_true, y_pred)

    Ntrue = np.unique(y_true).shape[0] # number of clusters
    Npred = np.unique(y_pred).shape[0]

    return {'ari':ari, 'ami':ami, 'vm':vm, 'homo':homo, 'complete':complt, 'Ntrue':Ntrue, 'Npred':Npred}


def binary_cluster_accuracy(y_true, y_pred):
    vm = metrics.v_measure_score(y_true, y_pred)
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    balacc = metrics.balanced_accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    homo = metrics.homogeneity_score(y_true, y_pred)
    complt = metrics.completeness_score(y_true, y_pred)

    # Nb = np.unique(y_pred).shape[0] # number of clusters

    return {'ari':ari, 'ami':ami, 'vm':vm, 'homo':homo, 'complete':complt, 'acc':acc, 'balacc':balacc,
                	    'f1':f1, 'Cpred': int(y_pred.sum()), 'Ctrue': int(y_true.sum())}

def per_cluster_accuracy(y_true, y_pred):
    df = []
    for i in np.unique(y_true):
        # find label of cluster containing most elements of the true cluster
        v, cc = np.unique(y_pred[y_true==i], return_counts=True)
        idx = v[np.argmax(cc)]
        # compare true cluster with predicted cluster
        perf = binary_cluster_accuracy(y_true==i, y_pred==idx)
        perf['y_true_idx'] = i
        perf['y_pred_idx'] = idx
        df.append(perf)
    return df




def triangle_participation_ratio(g):
    """calculate triad particpant ratio for a graph.
    TPR is the fraction of nodes in g that belong in a triad.

    Args:
        g (ig.Graph): graph to find 

    Returns:
        float: fraction of nodes in a triad
    """
    vintriad = 0
    for v in g.vs:
        v_nbrs = g.neighbors(v.index)
        vs = set(v_nbrs) - {v.index}
        gen_degree = Counter(len(vs & (set(g.neighbors(w)) - {w})) for w in vs)
        ntriangle = sum(k*val for k, val in gen_degree.items())
        if ntriangle:
            vintriad +=1
    tp_ratio = vintriad/g.vcount()
    return tp_ratio

def avg_triangle_participation_ratio(g,y):
    tprs = []
    nlist = np.arange(y.shape[0])
    for i in np.unique(y):
        nodes = nlist[y==i]
        gs = g.subgraph(nodes)
        tpr = triangle_participation_ratio(gs)
        tprs.append(tpr)
    
    return np.array(tprs).mean()

def conductance(g, nodes):
    gs = g.subgraph(nodes)
    eint = sum(gs.degree())
    total = sum(g.degree(nodes))

    if not total: # if total 0 set cond to nan
        cond = np.nan
    else:
        cond = 1 - eint/total

    return cond

def avg_conductance(g, y):
    cc = []
    nlist = np.arange(y.shape[0])
    for i in np.unique(y):
        nodes = nlist[y==i]
        cond = conductance(g, nodes)
        cc.append(cond)
        
    return np.array(cc).mean()

def avg_density(g, y):
    ds = []
    nlist = np.arange(y.shape[0])
    for i in np.unique(y):
        nodes = nlist[y==i]
        gs = g.subgraph(nodes)
        dens = gs.density()
        ds.append(dens)
    return np.array(ds).mean()

def separability(g, nodes):
    gs = g.subgraph(nodes)
    internal = gs.ecount()
    total_degree = sum(g.degree(nodes))
    external = total_degree - 2*internal

    if not external: # if 0 division set as nan
        sep = np.nan
    else:
        sep = internal/external

    return sep

def avg_separability(g, y):
    seps = []
    nlist = np.arange(y.shape[0])
    for i in np.unique(y):
        nodes = nlist[y==i]
        sep = separability(g, nodes)
        seps.append(sep)
    return np.array(seps).mean()

def avg_clustering(g, y):
    vals = []
    nlist = np.arange(y.shape[0])
    for i in np.unique(y):
        nodes = nlist[y==i]
        ccoef = g.transitivity_local_undirected(nodes)
        # vals.append(np.nanmean(ccoef))
        vals.append(nanmean(ccoef, allnanvalue=0)) # if all nan's replace with 0
    return np.mean(vals)

def cluster_quality(g, y):
    """Return stats describing cluster quality 
    - conductance
    - modularity
    - triad participation ratio
    and "community goodness"
    - separability
    - density
    - clustering coefficient
    note: ideas of cluster quality and community goodness 
    taken from https://dl.acm.org/doi/abs/10.1145/2350190.2350193

    Args:
        g (_type_): _description_
        y (_type_): _description_
    """
    tpr = avg_triangle_participation_ratio(g, y)
    mod = g.modularity(y)
    cond = avg_conductance(g, y)
    dens = avg_density(g, y)
    sep = avg_separability(g, y)
    cc = avg_clustering(g, y)

    ddict= {'mod':mod, 'cond':cond, 'tpr':tpr,
            'sep':sep, 'density':dens, 'cc':cc}
    return ddict

def single_cluster_quality(g, nodes):
    gs = g.subgraph(nodes)

    tpr = triangle_participation_ratio(gs)
    cond = conductance(g, nodes)
    sep = separability(g, nodes)
    dens = gs.density()
    ccoef = g.transitivity_local_undirected(nodes)

    return {'cond':cond, 'tpr':tpr, 'sep':sep, 'density':dens, 'cc':ccoef}

def per_cluster_quality(g, y, suffix=''):
    df = []
    nlist = np.arange(y.shape[0])
    for i in np.unique(y):
        nodes = nlist[y==i]
        # binary_cluster_qualtiy
        perf = single_cluster_quality(g, nodes)
        mod = g.modularity(y==i)
        perf['mod'] = mod
        perf = {k+suffix:v for k,v in perf.items()}
        df.append(perf)
    return df