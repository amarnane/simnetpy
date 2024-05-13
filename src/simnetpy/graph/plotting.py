import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition as decomp


def check_community_order(y):
    mem = set()
    order = []
    for idx in y:
        if idx not in mem:
            order.append(idx)
            mem.add(idx)
    return order


def gen_layout(g, method, scale_factor, center):
    l = g.layout(layout=method)
    # l.fit_into(bbox)
    l.scale(scale_factor)
    l.center(center)
    return l


def cluster_graph_frame(g, attr, layout_alg="fr"):
    clstr = ig.VertexClustering.FromAttribute(g, attr)
    ggclst = clstr.cluster_graph(combine_edges=dict(weight="sum"))
    frame = ggclst.layout(layout_alg)
    return frame, clstr, ggclst


def layout_per_cluster(
    clstr,
    ggclst,
    centers,
    N,
    nclstr_factor=0.5,
    clstr_size_factor=0.5,
    node_layout_alg="fr",
    vs_id_attr="name",
):
    layouts = []

    for i, gg in enumerate(clstr.subgraphs()):
        # change scaling based on number nodes
        # factor = gg.vcount()/ N
        scale_factor = (
            nclstr_factor * 1 / (ggclst.vcount()) + clstr_size_factor * gg.vcount() / N
        )
        # scale_factor = factor/(ggclst.vcount())

        ll = gen_layout(
            gg, node_layout_alg, scale_factor=scale_factor, center=centers[i]
        )
        layouts.append({v: coord for v, coord in zip(gg.vs[vs_id_attr], ll)})

    # unpack layouts
    layout_dict = {}
    for dd in layouts:
        layout_dict = {**layout_dict, **dd}
    layout = [layout_dict[x] for x in sorted(layout_dict.keys(), key=lambda x: int(x))]
    return layout


def group_by_cluster_layout(
    g,
    y_group,
    nclstr_factor=0.1,
    clstr_size_factor=1,
    frame_layout_alg="fr",
    node_layout_alg="fr",
):
    N = g.vcount()
    g.vs["name"] = list(np.arange(N))
    g.vs["y_group"] = list(y_group)
    # g.vs['y_color'] = list(y_color)

    frame, clstr, ggclst = cluster_graph_frame(
        g, attr="y_group", layout_alg=frame_layout_alg
    )
    layout = layout_per_cluster(
        clstr,
        ggclst,
        centers=frame.coords,
        N=N,
        nclstr_factor=nclstr_factor,
        clstr_size_factor=clstr_size_factor,
        node_layout_alg=node_layout_alg,
    )

    return layout


def plot_by_cluster(
    g,
    y_color,
    y_group,
    nclstr_factor=0.1,
    clstr_size_factor=0.2,
    layout=None,
    markersize=5,
    scale_marker=False,
    edge_alpha=0.5,
    edge_width=0.5,
    node_alpha=0.3,
    ax=None,
    style_dict=None,
    **kwds,
):
    if layout is None:
        layout = group_by_cluster_layout(
            g, y_group, nclstr_factor=nclstr_factor, clstr_size_factor=clstr_size_factor
        )

    L = np.array(layout)

    if ax is None:
        fig, ax = plt.subplots(dpi=200)
    else:
        fig = None

    dd = np.array(g.degree()) + 1  # +1 to ensure isolated nodes appear on plot
    dd = dd / dd.max()
    dd = (dd**2) * (10 * markersize**2)

    nclstr = len(np.unique(y_group))
    for i, c in enumerate(np.unique(y_group)):
        idx = y_group == c
        c = [f"C{int(j)}" for j in y_color[idx]]
        if scale_marker:
            ax.scatter(
                x=L[idx, 0],
                y=L[idx, 1],
                c=c,
                s=dd[idx],
                alpha=node_alpha,
                marker=".",
                linewidths=0,
                **kwds,
            )
        else:
            ax.scatter(
                x=L[idx, 0],
                y=L[idx, 1],
                c=c,
                alpha=node_alpha,
                s=markersize**2,
                marker=".",
                linewidths=0,
                **kwds,
            )

    if style_dict is None:
        style_dict = {
            "edge_width": edge_width,
            "edge_color": "lightgrey",
        }

    style_dict["layout"] = layout
    ig.plot(g.plottable(), target=ax, **style_dict)
    # fix aspect ratio
    ax.set_aspect("auto")

    N = g.vcount()
    ecount = g.ecount()
    # remove igraph circles (we use scatter instead)
    [
        edge.set_alpha(edge_alpha)
        for edge in ax.get_children()[N + nclstr : N + nclstr + ecount]
    ]
    [vertex.remove() for vertex in ax.get_children()[nclstr : N + nclstr]]
    # change alpha of edges
    return fig, ax


def network_plot_cmap(
    g,
    X,
    c,
    markersize=3,
    scale_marker=False,
    edge_alpha=0.5,
    edge_width=0.5,
    node_alpha=0.3,
    style_dict=None,
    PCA=False,
    ax=None,
    **scatterkwds,
):
    # create new figure if user ax not provided
    if ax is None:
        fig, ax = plt.subplots(dpi=100)
    else:
        fig = None

    if PCA:
        pca = decomp.PCA(n_components=2)
        Z = pca.fit_transform(X)
    else:
        Z = X[:, :2]

    dd = np.array(g.degree()) + 1  # +1 to ensure isolated nodes appear on plot
    dd = dd / dd.max()
    dd = (dd**2) * (markersize**2)

    # zorder adjusts plotting order so nodes cover edges.
    if scale_marker:
        cbar = ax.scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            c=c,
            s=dd,
            alpha=node_alpha,
            marker=".",
            linewidths=0,
            zorder=10,
            **scatterkwds,
        )
    else:
        cbar = ax.scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            c=c,
            s=markersize**2,
            alpha=node_alpha,
            marker=".",
            linewidths=0,
            zorder=10,
            **scatterkwds,
        )

    if fig is not None:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(cbar, cax=cbar_ax)

    if style_dict is None:
        style_dict = {
            "edge_width": edge_width,
            "edge_color": "lightgrey",
        }

    if isinstance(Z, np.ndarray):
        coords = Z.tolist()
    else:
        coords = Z
    layout = ig.Layout(coords=coords)
    style_dict["layout"] = layout

    ig.plot(g.plottable(), target=ax, **style_dict)
    ax.set_aspect("auto")
    N = g.vcount()
    ecount = g.ecount()
    [edge.set_alpha(edge_alpha) for edge in ax.get_children()[N + 1 : N + 1 + ecount]]
    [vertex.remove() for vertex in ax.get_children()[1 : N + 1]]

    return fig, ax


def network_plot_col_by_cluster(
    g,
    X,
    y,
    markersize=3,
    min_markersize=1.5,
    scale_marker=False,
    edge_alpha=0.5,
    edge_width=0.5,
    node_alpha=0.3,
    style_dict=None,
    PCA=False,
    ax=None,
    **scatterkwds,
):
    # create new figure if user ax not provided
    if ax is None:
        fig, ax = plt.subplots(dpi=100)
    else:
        fig = None

    if PCA:
        pca = decomp.PCA(n_components=2)
        Z = pca.fit_transform(X)
    else:
        Z = X[:, :2]

    dd = np.array(g.degree()) + 1  # +1 to ensure isolated nodes appear on plot
    dd = dd / dd.max()
    dd = (dd**2) * (markersize**2) + min_markersize**2

    # zorder adjusts plotting order so nodes cover edges.
    nclstr = len(np.unique(y))
    for i, c in enumerate(np.unique(y)):
        idx = y == c

        if scale_marker:
            ax.scatter(
                x=Z[idx, 0],
                y=Z[idx, 1],
                c=f"C{i}",
                s=dd[idx],
                alpha=node_alpha,
                marker=".",
                linewidths=0,
                zorder=10,
                **scatterkwds,
            )
        else:
            ax.scatter(
                x=Z[idx, 0],
                y=Z[idx, 1],
                c=f"C{i}",
                s=markersize**2,
                alpha=node_alpha,
                marker=".",
                linewidths=0,
                zorder=10,
                **scatterkwds,
            )

    if style_dict is None:
        style_dict = {
            "edge_width": edge_width,
            "edge_color": "lightgrey",
        }

    if isinstance(Z, np.ndarray):
        coords = Z.tolist()
    else:
        coords = Z
    layout = ig.Layout(coords=coords)
    style_dict["layout"] = layout

    ig.plot(g.plottable(), target=ax, **style_dict)
    ax.set_aspect("auto")
    N = g.vcount()
    ecount = g.ecount()
    [edge.set_alpha(edge_alpha) for edge in ax.get_children()[N+nclstr:N+nclstr+ecount]]
    [vertex.remove() for vertex in ax.get_children()[nclstr:N+nclstr]]

    return fig, ax
