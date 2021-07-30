import numpy as np
import random

from collections import Counter
from graphs.base_graph import BaseGraph
from analysis.utils.metrics_utils import clean_labels
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from analysis.utils.metrics_utils import build_m
from analysis.utils.metrics_utils import build_full_m
from analysis.utils.metrics_utils import build_m_modified_ref
from analysis.utils.metrics_utils import kld
from analysis.utils.metrics_utils import kld_normalized
import analysis.metrics as _metric

"""
This module contains different metric function and can be extended to new ones.
method signature expected:
    def name(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
"""


def adjusted_rand_index(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the adjusted RandIndex of two clustered graphs,
        where only nodes of the second graph are considered.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: adjusted randIndex value
    """
    return metrics.adjusted_rand_score(*clean_labels(reference_graph.labels, graph.labels))


def purity(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the purity of two clustered graphs,
        where only nodes of the second graph are considered.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: purity value
    """
    contingency_matrix = metrics.cluster.contingency_matrix(*clean_labels(reference_graph.labels, graph.labels))
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def accuracy(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the accuracy with optimal mapping of two clustered graphs,
        where only nodes of the second graph are considered.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: accuracy value
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(*clean_labels(reference_graph.labels, graph.labels))

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


def inverse_jensen_shannon_distance(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the Jensen Shannon Distance between two clustered graphs

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: Jensen Shannon Distance value
    """
    # get community probability vec
    ref_cluster_prob = [com / reference_graph.get_number_nodes() for com in reference_graph.get_community_sizes()]
    g_cluster_prob = [com / graph.get_number_nodes() for com in graph.get_community_sizes()]

    # size them to same size
    for _ in range(len(g_cluster_prob), len(ref_cluster_prob)):
        g_cluster_prob.append(0.0)
    for _ in range(len(ref_cluster_prob), len(g_cluster_prob)):
        ref_cluster_prob.append(0.0)

    return 1 - jensenshannon(ref_cluster_prob, g_cluster_prob, base=2)


def jensen_shannon_divergence(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the Jensen Shannon Divergence between two clustered graphs

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: Jensen Shannon Distance value
    """
    return (1 - inverse_jensen_shannon_distance(reference_graph, graph, params))**2


def cluster_size_diff_stripped(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the inverse normalized sum of the difference between each graphs cluster sizes.
    Nodes not contained in the :reference_graph: are removed/stripped from the reference for this calculation.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: value of the metric
    """
    ref_cluster_sizes = []
    g_clusters_sizes = sorted(graph.get_community_sizes(), reverse=True)

    not_added_nodes = set(reference_graph.G.nodes()) - set(graph.G.nodes())

    for k, v in reference_graph.get_community_nodes().items():
        ref_cluster_sizes.append(len(v) - len(set(v).intersection(not_added_nodes)))

    # check lengths
    abs_diff = abs(len(ref_cluster_sizes) - len(g_clusters_sizes))
    if len(ref_cluster_sizes) < len(g_clusters_sizes):
        ref_cluster_sizes.extend([0] * abs_diff)
    else:
        g_clusters_sizes.extend([0] * abs_diff)

    num_clusters = len(ref_cluster_sizes)
    c_sum = 0
    for i in range(num_clusters):
        c_sum += abs(g_clusters_sizes[i] - ref_cluster_sizes[i])

    norm_factor = graph.get_number_nodes() * (1 + (num_clusters - 2) / num_clusters)
    norm_factor = norm_factor if norm_factor != 0 else 1

    return 1 - c_sum / norm_factor


def cluster_num_diff(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Returns the difference of numbers of clusters between two graphs (both have to be clustered).

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :return float: diff
    """
    return abs(reference_graph.get_number_communities() - graph.get_number_communities())


def cluster_size_difference(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Returns the sum of difference between clusteres of two graphs

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :return float: diff
    """
    ref_cluster_sizes = reference_graph.get_community_sizes()
    g_clusters_sizes = graph.get_community_sizes()

    abs_diff = abs(len(ref_cluster_sizes) - len(g_clusters_sizes))
    if len(ref_cluster_sizes) < len(g_clusters_sizes):
        ref_cluster_sizes.extend([0] * abs_diff)
    else:
        g_clusters_sizes.extend([0] * abs_diff)

    _sum = 0
    for i in range(len(ref_cluster_sizes)):
        _sum += abs(ref_cluster_sizes[i] - g_clusters_sizes[i])

    return _sum


def jsd_approximation_entropy(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on entropy.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param params: containing the threshold value
        :return float: jsd based on entropy
    """
    return _metric.entropy_approximation(build_m(reference_graph, graph), params) - (_metric.entropy_approximation(reference_graph, params) + _metric.entropy_approximation(graph, params)) / 2


def jsd_approximation_entropy_normalized(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on normalized entropy.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param params: containing the threshold value
        :return float: jsd based on entropy
    """
    return _metric.entropy_approximation_normalized(build_m(reference_graph, graph), params) - (_metric.entropy_approximation_normalized(reference_graph, params) + _metric.entropy_approximation_normalized(graph, params)) / 2


def jsd_approximation_kld(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on KLD.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param params: containing the threshold value
        :return float: jsd based on entropy
    """
    threshold = params.get('threshold', 2.5)

    m, rf = build_m_modified_ref(reference_graph, graph)
    return (kld(rf, m, threshold) + kld(graph, m, threshold)) / 2


def jsd_approximation_kld_normalized(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on normalized KLD.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param params: containing the threshold value
        :return float: jsd based on entropy
    """
    threshold = params.get('threshold', 2.5)

    m, rf = build_m_modified_ref(reference_graph, graph)
    return (kld_normalized(rf, m, threshold) + kld_normalized(graph, m, threshold)) / 2


def jsd_approximation_apd(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on APD.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param k=sample_size: sample size to take from the graphs
        :return float: jsd based on entropy
    """
    return _metric.apd(build_m(reference_graph, graph), params) - (_metric.apd(reference_graph, params) + _metric.apd(graph, params)) / 2


def jsd_approximation_apd_normalized(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on normalized APD.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param params k=sample_size: sample size to take from the graphs and
        :param params k=norm_factor: normalization factor
        :return float: jsd based on entropy
    """
    return _metric.apd_normalized(build_m(reference_graph, graph), params) - (_metric.apd(reference_graph, params) + _metric.apd(graph, params)) / 2


def jsd_approximation_hpd(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on HPD.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param k=sample_size: sample size to take from the graphs
        :return float: jsd based on entropy
    """
    return _metric.hpd(build_m(reference_graph, graph), params) - (_metric.hpd(reference_graph, params) + _metric.hpd(graph, params)) / 2


def jsd_approximation_hpd_normalized(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the JSD based on normalized HPD.

    Args:
        :param reference_graph: first graph
        :param graph: second graph
        :param k=sample_size: sample size to take from the graphs
        :param params k=norm_factor: normalization factor
        :return float: jsd based on entropy
    """
    return _metric.hpd_normalized(build_m(reference_graph, graph), params) - (_metric.hpd_normalized(reference_graph, params) + _metric.hpd_normalized(graph, params)) / 2


def stripped_entropy(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy of the reference graph with nodes removed, which are not present in graph

    Args:
        :param reference_graph: graph to strip and calculate the entropy on
        :param graph: holds nodes to be used in the entropy calculation
        :return float: the stripped entropy of the reference graph
    """
    ref_cluster_sizes = []
    not_added_nodes = set(reference_graph.G.nodes()) - set(graph.G.nodes())

    for k, v in reference_graph.get_community_nodes().items():
        ref_cluster_sizes.append(len(v) - len(set(v).intersection(not_added_nodes)))

    return entropy(ref_cluster_sizes, base=2)


def stripped_entropy_normalized(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy of the reference graph with nodes removed, which are not present in graph

    Args:
        :param reference_graph: graph to strip and calculate the entropy on
        :param graph: holds nodes to be used in the entropy calculation
        :return float: the stripped entropy of the reference graph
    """
    ref_cluster_sizes = []
    not_added_nodes = set(reference_graph.G.nodes()) - set(graph.G.nodes())

    for k, v in reference_graph.get_community_nodes().items():
        ref_cluster_sizes.append(len(v) - len(set(v).intersection(not_added_nodes)))

    return entropy(ref_cluster_sizes, base=2) / np.log2(len(ref_cluster_sizes))


def distance_h_h(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy distance between two graphs, where the clustering is known.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: value of the metric
    """
    return abs(entropy(graph.get_community_sizes(), base=2) - entropy(graph.get_community_sizes(), base=2))


def distance_h_hn(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy distance between clustered and unclustered graph.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :param params: dictionary holding the threshold value
        :returns float: value of the metric
    """
    return abs(entropy(graph.get_community_sizes(), base=2) - _metric.entropy_approximation(graph, params))


def distance_h_apd(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy-apd distance between two graphs.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :param params: dictionary holding the sample size
        :returns float: value of the metric
    """
    return abs(entropy(graph.get_community_sizes(), base=2) - _metric.apd(graph, params))


def distance_h_hpd(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy-hpd distance between two graphs.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :param params: dictionary holding the sample size
        :returns float: value of the metric
    """
    return abs(entropy(graph.get_community_sizes(), base=2) - _metric.hpd(graph, params))


def distance_stripped_h_h(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy distance between two graphs, where the clustering is known and the
        reference graph is stripped.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :returns float: value of the metric
    """
    return abs(stripped_entropy(reference_graph, graph, {}) - entropy(graph.get_community_sizes(), base=2))


def distance_stripped_h_hn(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the entropy distance between stripped clustered and unclustered graph.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :param params: dictionary holding the threshold value
        :returns float: value of the metric
    """
    return abs(stripped_entropy(reference_graph, graph, {}) - _metric.entropy_approximation(graph, params))


def distance_stripped_h_apd(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the stripped entropy-apd distance between two graphs.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :param params: dictionary holding the sample size
        :returns float: value of the metric
    """
    return abs(stripped_entropy(reference_graph, graph, {}) - _metric.apd(graph, params))


def distance_stripped_h_hpd(reference_graph: BaseGraph, graph: BaseGraph, params: dict) -> float:
    """
    Calculates the stripped entropy-hpd distance between two graphs.

    Args:
        :param reference_graph: reference graph
        :param graph: graph to check against reference
        :param params: dictionary holding the sample size
        :returns float: value of the metric
    """
    return abs(stripped_entropy(reference_graph, graph, {}) - _metric.hpd(graph, params))
