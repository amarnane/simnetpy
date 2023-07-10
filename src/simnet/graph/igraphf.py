import time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import igraph as ig
# from ..plotting import plot_degree_dist




def convert_adj_2_igraph(A):
    A = sp.triu(A).tocoo()
    E = np.vstack((A.row, A.col)).T
    E=list(map(tuple, E))

    # X = {var: row for var, row in enumerate(g.x.T)}
    return ig.Graph(E)

def find_second_cc(cc):
    # cc = np.array(g2.clusters().membership)
    C = np.vstack(np.unique(cc, return_counts=True))
    assert C[1, :].max() == C[1,0], "LCC is not 0th component"
    cc2 = C[:,(C[1, :] > 1) & (C[1,:] < C[1,0])] # find values lower than max but greater than 1 (might be empty?)
    return tuple(cc2[:,cc2[1,:].argmax()]) # returns index and size

def find_first_second_largest_cc(cc):
    C = np.vstack(np.unique(cc, return_counts=True))
    cmax = C[:,C[1,:].argmax()]
    cc2 = C[:,(C[1, :] > 1) & (C[1,:] < cmax[1])] # find values lower than max but greater than 1 (might be empty?)
    try:
        cc2 = cc2[:,cc2[1,:].argmax()]
    except ValueError:
        return cmax[0], None # if no second component return None.

    return cmax[0], cc2[0] # returns index for largest and 2nd largest components

def graph_stats(gg):
    ddict = {}
    ddict['density'] = gg.density()
    ddict['n'] = gg.vcount()
    ddict['E'] = gg.ecount()
    ddict['diameter'] = gg.diameter()
    ddict['globalcc'] = gg.transitivity_undirected(mode='zero')
    ddict['avglocalcc'] = gg.transitivity_avglocal_undirected(mode='zero')
    ddict['assortivity'] = gg.assortativity_degree(directed=False)
    ddict['avg_path_length'] = gg.average_path_length(directed=False)
    dd = np.array(gg.degree())
    ddict['avg_degree'] = dd.mean()
    ddict['median_degree'] = np.median(dd)
    ddict['max_degree'] = dd.max()


    return ddict

def component_info(gg):
    ddict = {}
    ddict['E'] = gg.ecount()
    
    cc = gg.clusters()
    ddict['Nc'] = len(cc)

    idxmax, idx2 = find_first_second_largest_cc(cc.membership)

    gc = cc.subgraph(idx=idxmax)
    ddict['cc_max'] = graph_stats(gc)
    
    ccmem = np.array(cc.membership)
    ddict['cc_max_mem'] = (ccmem==idxmax)

    if idx2 is not None:
        gc2 = cc.subgraph(idx=idx2)
        ddict['cc2'] = graph_stats(gc2)
    else:
        ddict['cc2'] = None
    return ddict


def compare_clusters(c1, c2):
    ddict = {}
    for metric in ['vi', 'nmi', 'split-join', 'rand', 'adjusted_rand']:
        try:
            dist = ig.compare_communities(c1, c2, method=metric)
        except:
            dist = np.nan
        ddict[metric] = dist
    return ddict
    
def cluster_comparison(clist):
    ddict = {}
    for i, c1 in enumerate(clist):
        for j, c2 in enumerate(clist[i+1:],i+1):
            # print(i,j)
            ddict[(i,j)] = compare_clusters(c1, c2)
    return ddict


# # NOTE: Still have issue here where passing functions doesn't really work.
# # We rely on gg being defined. And being the same as the inputted graph.
# # not sure how to call methods without having the method defined.
# # try using getattr?
# def run_communities(gg, community_funcs=None):
#     if community_funcs is None:
#         community_funcs = {'fastgreedy': {'f': gg.community_fastgreedy, 'kwargs':{'weights':None}}, 
#         'infomap': {'f':gg.community_infomap, 'kwargs':{'trials':10}},
#         # 'leading_eigen_naive': {'f':gg.community_leading_eigenvector_naive, 'kwargs':{'return_merges':False}}, # might be issue. not sure if kwarg valid for this as well as naive
#         'leading_eigen': {'f': gg.community_leading_eigenvector, 'kwargs':{'weights': None, 'arpack_options':ig._igraph.ARPACKOptions(maxiter=30000)}},
#         'label_prop': {'f':gg.community_label_propagation, 'kwargs':{'weights':None}},
#         'multilevel': {'f':gg.community_multilevel, 'kwargs':{'return_levels':False}},
#         # 'edge_betweenness': {'f':gg.community_edge_betweenness, 'kwargs':{'directed':False}},
#         # 'spinglass':{'f':gg.community_spinglass, 'kwargs':{'weights':None}},
#         'walktrap':{'f':gg.community_walktrap, 'kwargs':{'steps':4}},
#         'leiden':{'f':gg.community_leiden,'kwargs':{'n_iterations':-1, 'objective_function':'modularity', 'resolution_parameter':1.0}} # -1 so runs until converges, 'cpm' is variant of modularity,
#         }
#     ddict = {}
#     for k, fdict in community_funcs.items():
#         print(f'On func: {k}')

#         # call detection method with arguments as specified in community_func dict
#         f = fdict['f']
#         kwargs = fdict['kwargs']
    
#         start = time.time()
#         try:
#             clust = f(**kwargs)
#         except Exception as e:
#             print(e)
#             print(k)
#             clust=None
#             ddict[k] = {'size':np.nan, 'modularity':np.nan, 'clustering':None, 'time':np.nan}
#             continue

#         end = time.time()

#         # some methods return a dendrogram
#         if isinstance(clust, ig.clustering.VertexDendrogram):
#             print(clust.summary())
#             clust = clust.as_clustering()

#         # stats to store
#         num = len(clust)
#         mod = clust.modularity
#         ddict[k] = {'size':num, 'modularity':mod, 'clustering':clust, 'time':(end - start)}
#     return ddict


def ig_graphinfo(gg, community_funcs=None):
    ddict = {}
    component_stats = component_info(gg)    
    ddict['component_stats'] = component_stats

    ccmax = list(np.array(gg.vs.indices)[component_stats['cc_max_mem']])
    gc = gg.induced_subgraph(ccmax)

    # communities = run_communities(gc, community_funcs=community_funcs)
    # ddict['community_stats'] = communities

    clist = [ddict['clustering'] for k, ddict in communities.items()]
    clustercomp = cluster_comparison(clist)

    ddict['cluster_comp'] = clustercomp

    ddict['gg'] = gg
    ddict['gc'] = gc
    return ddict


def find_threshold_graph_stats(A):
    gg = convert_adj_2_igraph(A)
    stats = ig_graphinfo(gg)
    return stats



class Igraph(ig.Graph):
    def __init__(self, *args,**kwds):
        X = kwds.pop('X', None) # check if X passed as argument

        if isinstance(X, pd.DataFrame):
            X = X.to_dict(orient='list')

        super().__init__(*args,**kwds)

    def graph_stats(self):
        ddict = {}
        ddict['density'] = self.density()
        ddict['n'] = self.vcount()
        ddict['E'] = self.ecount()
        C = self.component_sizes(pcent=False)
        ddict['Nc'] = C.shape[1]
        ddict['ncmax'] = C[1,0]
        ddict['diameter'] = self.diameter()
        ddict['globalcc'] = self.transitivity_undirected(mode='zero')
        ddict['avglocalcc'] = self.transitivity_avglocal_undirected(mode='zero')
        ddict['assortivity'] = self.assortativity_degree(directed=False)
        ddict['avg_path_length'] = self.average_path_length(directed=False)
        dd = np.array(self.degree())
        ddict['avg_degree'] = dd.mean()
        ddict['median_degree'] = np.median(dd)
        ddict['max_degree'] = dd.max()

        return ddict


    def component_stats(self):
        ddict = {}
        ddict['E'] = self.ecount()
        
        cc = self.connected_components()
        ddict['Nc'] = len(cc)

        idxmax, idx2 = self.find_first_second_largest_cc(cc.membership)

        gc = cc.subgraph(idx=idxmax)
        gc = self.from_igraph(gc)
        ddict['cc_max'] = gc.graph_stats()
        
        ccmem = np.array(cc.membership)
        ddict['cc_max_mem'] = (ccmem==idxmax)

        if idx2 is not None:
            gc2 = cc.subgraph(idx=idx2)
            gc2 = self.from_igraph(gc2)
            ddict['cc2'] = gc2.graph_stats()
        else:
            ddict['cc2'] = None
        return ddict
    
    def component_sizes(self, cc=None, pcent=True):
        if cc is None:
            cc = self.connected_components()

        C = np.vstack(np.unique(cc.membership, return_counts=True))
        C = C[:,np.argsort(C[1,:])[::-1]] # sort so largest is first
        if pcent:
            C = C.astype('float64')
            C[1,:] = C[1,:]/len(cc.membership)

        return C

    def large_components(self, large_comp_cutoff=0.1, cc=None, return_idx=False):
        """Find components in network greater than large_comp_cutoff.

        Args:
            large_comp_cutoff (float/int, optional): Cutoff to be considered a large component. 
                        If int is number of nodes, if float is percentage of total nodes. 
                        note decimal percentage so 33% cutoff would be 0.33.Defaults to 0.1
            cc (ig.VertexClustering, optional): precalculated components/clustering. 
                        Typically output of g.connected_components. Defaults to None.
            return_idx (bool, optional): flag wether to return indexs of components/clusters. Defaults to False.

        Returns:
            list/tuple of lists: if return idx returns tuple containing list of large component subgraphs and list of indexes
                                otherwise returns just list of component subgraphs
        """
        if cc is None:
            cc = self.connected_components()

        if isinstance(large_comp_cutoff, float):
            pcent = True
        elif isinstance(large_comp_cutoff, int):
            pcent = False
        else:
            raise ValueError('large_comp_cutoff should be node number or decimal percentage (i.e. 33% cutoff should be 0.33)')

        C = self.component_sizes(cc, pcent=pcent)

        large_comps = C[0,C[1,:] >= large_comp_cutoff]
        components = [cc.subgraph(idx) for idx in large_comps]

        return (components, large_comps)  if return_idx else components



    def find_first_second_largest_cc(self, cc=None):
        if cc is None:
            cc = self.connected_components()
            cc = cc.membership
    
        C = np.vstack(np.unique(cc, return_counts=True))
        cmax = C[:,C[1,:].argmax()]
        cc2 = C[:,(C[1, :] > 1) & (C[1,:] < cmax[1])] # find values lower than max but greater than 1 (might be empty?)
        try:
            cc2 = cc2[:,cc2[1,:].argmax()]
        except ValueError:
            return cmax[0], None # if no second component return None.

        return cmax[0], cc2[0] # returns index for largest and 2nd largest components

    def get_comp(self, idx, cc=None):
        if cc is None:
            cc = self.connected_components()
        
        gc = cc.subgraph(idx=idx)

        return self.from_igraph(gc)

    def max_component(self):
        cc = self.connected_components()
        idxmax, idx2 = self.find_first_second_largest_cc(cc.membership)
        return self.get_comp(idx=idxmax, cc=cc)

    def get_2ndmax_comp(self):
        cc = self.connected_components()
        idxmax, idx2 = self.find_first_second_largest_cc(cc.membership)
        if idx2 is None:
            raise ValueError('No 2nd largest component')
        return self.get_comp(idx=idx2, cc=cc)

    def get_X(self):
        X = []
        for k in self.vs.attributes():
            v = self.vs[k]
            X.append(v)
        X = np.array(X).T # Transpose to have shape n x d
        return X

    # def plot_degree_dist(self, logbinsize=0.1, LOG_ONLY=False):
    #     degree = self.degree()
    #     # plot_degree_dist(degree_dist=degree, logbinsize=logbinsize, LOG_ONLY=LOG_ONLY)

    def plottable(self, mode='undirected'):
        A = self.get_adjacency_sparse()
        return ig.Graph().Adjacency(A, mode=mode)

    @classmethod
    def from_igraph(cls, gg, *args, **kwds):
        return cls(*args, **kwds) + gg