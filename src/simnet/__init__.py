from .version import __version__

from .clustering import (
    sbm_clustering,
    spectral_clustering,
    leiden_clustering,
    cluster_quality,
    cluster_accuracy,
    binary_cluster_accuracy,
    per_cluster_accuracy,
)

from .datasets import (
    mixed_multi_guassian,
    mixed_categorical_clusters,
)

from .similarity import (
    pairwise_sim,
    multi_modal_similarity,
    probability_matrix, 
    network_from_sim_mat,
    sparsify_sim_matrix,
)



from .graph import Igraph
from .utils import create_dirs_on_path

from . import datasets
from . import utils
from . import plotting


