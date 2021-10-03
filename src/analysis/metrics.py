import numpy as np
import random

from collections import Counter
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import adjusted_rand_score as ari

from graphs.base_graph import BaseGraph
from analysis.utils.metrics_utils import random_sample_cluster, pertubate, gen_labels


def entropy_clustered(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy of an clustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :returns float: value of the metric
    """
    return entropy(graph.get_community_sizes(), base=2)


def entropy_clustered_normalized(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy of an clustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :returns float: value of the metric
    """
    if graph.get_number_communities() > 1:
        return entropy(graph.get_community_sizes(), base=2) / np.log2(graph.get_number_communities())
    return entropy(graph.get_community_sizes(), base=2)


def entropy_approximation(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the approximate entropy of an unclustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :param threshold: which edges to consider
        :returns float: value of the metric
    """
    threshold = params.get('threshold', 2.5)

    num_nodes = graph.get_number_nodes()
    if num_nodes == 0:
        return 0
    node_num_edges_over_threshold = Counter([node for k, v in graph.get_weight_edge().items() if k >= threshold for t in v for node in t])

    s_sum = 0
    for i in graph.G.nodes():
        s_sum += np.log2((1 + node_num_edges_over_threshold.get(i, 0)) / num_nodes)

    return -(s_sum / num_nodes)


def entropy_approximation_normalized(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the approximate entropy of an unclustered graph.
    Args:
        :param graph: on which to performe the evaluation
        :param threshold: which edges to consider
        :returns float: value of the metric
    """
    threshold = params.get('threshold', 2.5)

    num_nodes = graph.get_number_nodes()
    if num_nodes == 0:
        return 0
    node_num_edges_over_threshold = Counter([node for k, v in graph.get_weight_edge().items() if k >= threshold for t in v for node in t])

    s_sum = 0
    for i in graph.G.nodes():
        s_sum += np.log2((1 + node_num_edges_over_threshold.get(i, 0)) / num_nodes)

    h = -(s_sum / num_nodes)
    return h / np.log2(graph.get_number_nodes()) if graph.get_number_nodes() > 1 else h


def invers_entropy_distance(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse entropy distance between the approximate entropy and real entropy is used.

    Args:
        :param graph: graph to check against reference
        :threshold: which edge value should be regarded
        :returns float: value of the metric
    """
    return 1 - abs(entropy_clustered(graph, params) / np.log2(graph.get_number_communities()) - entropy_approximation(graph, params) / np.log2(graph.get_number_nodes()))


def apd(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the APD (Average Pointwise Distance) of a graph (edge weights describing the distance).

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :return float: apd of the sample
    """
    sample_size = params.get('sample_size', 100)
    sampled_edge_list = []

    for _ in range(sample_size):
        u, v = sorted(random.sample(graph.G.nodes(), 2))
        if graph.get_edge(u, v) is not None:
            sampled_edge_list.append(graph.get_edge(u, v))

    if len(sampled_edge_list) == 0:
        return 0
    return sum(sampled_edge_list) / len(sampled_edge_list)


def apd_normalized(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the normalized APD (Average Pointwise Distance).

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :param norm_factor: normalization factor
        :return float: apd of the sample
    """
    norm_factor = params.get('norm_factor', 4)

    return apd(graph, params) / norm_factor


def hpd(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy of the sampled pointwise distribution of edge weights.

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :return float: hpd of the sample
    """
    sample_size = params.get('sample_size', 100)
    sampled_edge_list = []

    for _ in range(sample_size):
        u, v = sorted(random.sample(graph.G.nodes(), 2))
        if graph.get_edge(u, v) is not None:
            sampled_edge_list.append(graph.get_edge(u, v))

    count_edges = [v for k, v in Counter(sampled_edge_list).items()]

    if len(count_edges) == 0:
        return 0
    return entropy(count_edges, base=2)


def hpd_normalized(graph: BaseGraph, params: dict) -> float:
    """
    Calculates the normalized entropy of the sampled pointwise distribution of edge weights.

    Args:
        :param graph: graph on which to calculate
        :param sample_size: the sample size to take from the graph
        :param norm_factor: normalization factor
        :return float: hpd of the sample
    """
    norm_factor = params.get('norm_factor', 4)

    return hpd(graph, params) / np.log2(norm_factor)


def cluster_number(graph: BaseGraph, params: dict) -> float:
    """Returns the number of clusters of a graph.
    Note: This is just for completness sake here.

    Args:
        graph (BaseGraph): Graph on which to take the number from
        params (dict): -

    Returns:
        float: number of clusters
    """
    return graph.get_number_communities()


def bootstraping_jsd(graph: BaseGraph, params: dict) -> float:
    """Calculates the bootstraped jsd confidence interval and returns the upper bound.
    This is equivalent to the stopping criterion, but returns the actual value of the upper percentile.

    Args:
        graph (BaseGraph): Graph on which to calculate the confidence interval
        params (dict): should contain {rounds, sample_size, alpha}

    Returns:
        float: returns the upper percentile
    """

    rounds = params.get('rounds', 30)
    assert type(rounds) == int

    sample_size = params.get('sample_size', 150)
    assert type(sample_size) == int

    alpha = params.get('alpha', 0.95)
    assert type(alpha) == float

    stats = []
    for _ in range(rounds):
        stats.append(jensenshannon(graph.get_community_sizes(), random_sample_cluster(graph, sample_size), base=2)**2)

    stats.sort()
    percentile = np.percentile(stats, [((1.0 - alpha) / 2.0) * 100, (alpha + ((1.0 - alpha) / 2.0)) * 100])
    return percentile[1]


def bootstraping_perturbation_ari(graph: BaseGraph, params: dict) -> float:
    """Calculates the robustness of the given graph, based on Gambettes method.
    This is equivalent to the stopping criterion, but returns the actual value of the mean.


    Args:
        graph (BaseGraph): Graph on which to calculate the mean
        params (dict): should contain {range, share, rounds, clustering_func, clustering_params}

    Returns:
        float: mean ARI score
    """
    rounds = params.get('rounds', 10)
    assert type(rounds) == int

    _range = params.get('range', (1, 4))
    assert type(_range) == tuple

    share = params.get('share', 0.1)
    assert type(share) == float

    clustering_func = params.get('clustering_func', None)
    assert callable(clustering_func)

    clustering_params = params.get('clustering_params', {})
    assert type(clustering_params) == dict

    stats = []
    for _ in range(rounds):
        new_graph = pertubate(graph, _range, share)
        cl = clustering_func(new_graph, clustering_params)
        new_graph.update_community_nodes_membership(cl)
        stats.append(ari(gen_labels(graph), gen_labels(new_graph)))

    return np.nanmean(stats)
