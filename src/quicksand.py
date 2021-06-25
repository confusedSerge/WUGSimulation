from import_scripts.wic_data_to_dict import gen_node_edge_dictionaries
from graphs.wic_graph import WICGraph
from simulation.clustering.clustering_strategy import new_correlation_clustering
from visualization.graph_visualization import draw_graph_graphviz as draw

# from import_scripts.input_generator import input_json_generator

# input_json_generator('next', 'dev.wu_1', 'ADJ', 'data/nicolay_data/test_data/dev.wu_1.txt', 'data/nicolay_data/test_data/dev.wu_1.input')

input_file = 'data/nicolay_data/test_data/dev.wu_1.input'
score_file = 'data/nicolay_data/test_data/dev.wu_1.scores'
tag_file = 'data/nicolay_data/test_data/dev.wu_1'

wic = WICGraph()
wic.build_nx_graph_from_dicts(*gen_node_edge_dictionaries(input_file, score_file, tag_file), 'score1')

# Testing stuff
wic.add_new_weight_dict('norm', weight_modifier=lambda x: x * 3 + 1)
wic.add_new_weight_dict('shifted', weight_modifier=lambda x: x * 3 - 1.5)

clusters = new_correlation_clustering(wic, {'weights': 'shifted', 'max_attempts': 10, 'max_iters': 10, 'split_flag': False})

wic.update_community_nodes_membership(clusters)

edge_weight: dict = wic.get_edge_weight()
weight_edge = dict()

for k, v in edge_weight.items():
    if weight_edge.get(round(v * 3 + 1), None) == None:
        weight_edge[round(v * 3 + 1)] = []
    weight_edge[round(v * 3 + 1)].append(k)

wic.G.graph['weight_edge'] = weight_edge

draw(wic, "WIC Test Graph")