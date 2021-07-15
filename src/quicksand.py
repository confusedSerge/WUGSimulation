import networkx

G = networkx.Graph()
G.add_weighted_edges_from([(0, 1, 10)])
print(G.get_edge_data(0, 1)['weight'])