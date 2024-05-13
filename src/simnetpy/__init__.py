# from .version import __version__
__all__ = [
    "graph",
    "datasets",
    "clustering",
    "similarity",
    "plotting",
    "utils",
    "pairwise_sim",
    "probability_matrix",
    "multi_modal_similarity",
    "network_from_sim_mat",
    "sparsify_sim_matrix",
    "mixed_multi_numeric",
    "mixed_categorical_clusters",
    "leiden_clustering",
    "sbm_clustering",
    "spectral_clustering",
    "cluster_accuracy",
    "cluster_quality",
    "per_cluster_accuracy",
    "binary_cluster_accuracy",
]

from . import clustering, datasets, graph, plotting, similarity, utils
from .clustering import (
    binary_cluster_accuracy,
    cluster_accuracy,
    cluster_quality,
    leiden_clustering,
    per_cluster_accuracy,
    sbm_clustering,
    spectral_clustering,
)
from .datasets import (
    mixed_categorical_clusters,
    mixed_multi_numeric,
)
from .graph import Igraph
from .similarity import (
    multi_modal_similarity,
    network_from_sim_mat,
    pairwise_sim,
    probability_matrix,
    sparsify_sim_matrix,
)

# from .utils import create_dirs_on_path, set_science_style, save_mpl_figure


# from . import datasets
# from . import utils
# from . import plotting
