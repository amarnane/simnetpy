import igraph as ig
import numpy as np


def gen_layout(g, method, bbox, factor, center):
    l = g.layout(layout=method)
    l.fit_into(bbox)
    l.scale(factor)
    l.center(center)
    return l
    

def cluster_layout(g, attr, vs_id_attr='name', layout='fr', bbox=(300,300)):

    cluster = ig.VertexClustering.FromAttribute(g, attr)
    ggclst = cluster.cluster_graph(combine_edges=dict(weight="sum"))
    frame = ggclst.layout(layout)
    frame.fit_into(bbox)
    
    N = g.vcount()
    layouts = []
    for i, gg in enumerate(cluster.subgraphs()):
        # change scaling based on number nodes
        factor = gg.vcount()/ N
        ll = gen_layout(gg, 'fr', bbox=bbox, factor=factor, center=frame.coords[i])
        layouts.append({v: coord for v, coord in zip(gg.vs[vs_id_attr],ll)})

    # Combine Node ID layout mapping
    layout_dict = {}
    for dd in layouts:
        layout_dict = {**layout_dict, **dd}

    # layout = [x[1] for x in sorted(layout_dict.items(), key=lambda x: int(x[0]))]

    return layout_dict
    
def filter_edges(g, pcentile=0.99):
    # find weights
    ww = np.array(g.es['weight'])
    cutoff=np.quantile(ww, pcentile)
    # filter edges using cutoff
    ee = {e.tuple: e['weight'] for e in  g.es.select(weight_ge=cutoff)}

    # find spanning tree so no isolated vertices
    gg = g.spanning_tree()

    M = gg.ecount() # start point for new edges in graph
    gg.add_edges(list(ee.keys())) # add most important edges
    gg.es[M:]['weight'] = list(ee.values()) # add their weights
    
    return gg

def plot_graph_unweighted_nodes_edges(g, layout=None, vcolors='#57a773', ewidth = 2, vsize = 10, bbox=(500,300), margin=20):
    visual_style={}
    colorset = ['#57a773', '#08b2e3', '#ffe900', '#ee6352', '#484d6d', '#efe9f4', '#d68fd6', '#6665dd', '#473bf0', '#9368b7', '#695958']

    visual_style["vertex_color"] = vcolors
    # visual_style["vertex_label"] = g.vs['name']
    visual_style['vertex_label_dist'] = -1.5
    visual_style['edge_color'] = '#b4b4b4'

    # visual_style["edge_width"] = gen_edge_width(g.es['weight'], ewidthfactor)
    # visual_style["vertex_size"] = [vsizefactor*x for x in g.vs['size']]
    visual_style['vertex_size'] = vsize
    visual_style['edge_width'] = ewidth
    # visual_style["edge_width"] = [w/10 for w in ggclst.es["weight"]]
    # visual_style["edge_width"] = [0.1/w**2 for w in g.es["weight"]]


    visual_style["layout"] = layout
    visual_style["bbox"] = bbox
    visual_style["margin"] = margin
    return ig.plot(g, **visual_style)

def plot_graph_by_attr(g, attr='cluster', ewidth = 2, vsize = 10, bbox=(500,300), margin=20):
    cluster = ig.VertexClustering.FromAttribute(g, attr)
    
    layout_dict = cluster_layout(g, attr)
    layout = [layout_dict[x] for x in sorted(layout_dict.keys(), key=lambda x: int(x))]

    colorset = ['#57a773', '#08b2e3', '#ffe900', '#ee6352', '#484d6d', '#efe9f4', '#d68fd6', '#6665dd', '#473bf0', '#9368b7', '#695958']
    vcolors=[colorset[i%len(colorset)] for i in cluster.membership]
    plot = plot_graph_unweighted_nodes_edges(g, layout, vcolors=vcolors,ewidth=ewidth, vsize=vsize, bbox=bbox, margin=margin)
    return plot


def gen_edge_width(weights, factor):
    w = np.array(weights)
    w = factor*w/w.max()
    return w.squeeze()

def plot_graph_weighted_nodes_and_edges(g, layout=None, vcolors='#57a773', ewidthfactor = 20, vsizefactor = 120, bbox=(500,300), margin=20):
    if layout is None:
        layout = g.layout('fr')
        layout.fit_into(bbox)
    elif isinstance(layout, str):
        layout = g.layout(layout)
        layout.fit_into(bbox)

    visual_style={}
    colorset = ['#57a773', '#08b2e3', '#ffe900', '#ee6352', '#484d6d', '#efe9f4', '#d68fd6', '#6665dd', '#473bf0', '#9368b7', '#695958']

    visual_style["vertex_color"] = vcolors
    visual_style["vertex_label"] = g.vs['name']
    visual_style['vertex_label_dist'] = -1.5
    visual_style['edge_color'] = '#b4b4b4'

    visual_style["edge_width"] = gen_edge_width(g.es['weight'], ewidthfactor)
    visual_style["vertex_size"] = [vsizefactor*x for x in g.vs['size']]
    # visual_style["edge_width"] = [w/10 for w in ggclst.es["weight"]]
    # visual_style["edge_width"] = [0.1/w**2 for w in g.es["weight"]]


    visual_style["layout"] = layout
    visual_style["bbox"] = bbox
    visual_style["margin"] = margin
    
    # fig, ax = plt.subplots(figsize=(8,8))
    # ig.plot(g, **visual_style).save('temporary.png') 
    # display(Image(filename='temporary.png'))
    # os.remove('temporary.png')
    return ig.plot(g, **visual_style)

def create_cluster_graph(g, attr):
    cluster = ig.VertexClustering.FromAttribute(g, attr)
    ggclst = cluster.cluster_graph(combine_edges=dict(weight="sum"))

    memmap = {k:v for k,v in zip(cluster.membership, g.vs[attr])}
    ggclst.vs['name'] = [memmap[i] for i in ggclst.vs.indices]
    ggclst.vs['size'] = [x/cluster.n for x in cluster.sizes()]
    return ggclst

def plot_cluster_graph(g, attr='cluster', layout='fr', ewidthfactor = 20, vsizefactor = 120, bbox=(500,300), margin=20):
    if 'weight' not in g.es.attribute_names():
        g.es['weight'] = list(np.ones(g.ecount()))
    
    ggclst = create_cluster_graph(g, attr)
    
    colorset = ['#57a773', '#08b2e3', '#ffe900', '#ee6352', '#484d6d', '#efe9f4', '#d68fd6', '#6665dd', '#473bf0', '#9368b7', '#695958']
    vcolors=[colorset[i%len(colorset)] for i in ggclst.vs.indices]
    
    plot = plot_graph_weighted_nodes_and_edges(ggclst, layout=layout, vcolors=vcolors, ewidthfactor=ewidthfactor, vsizefactor=vsizefactor, bbox=bbox, margin=margin)
    return plot