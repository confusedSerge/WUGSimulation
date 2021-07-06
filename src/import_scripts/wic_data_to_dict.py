import json

def gen_node_edge_dictionaries(input_file: str, score_file: str, tag_file: str):
    # load and build dict
    edge_info = _load_files_as_dict(input_file)
    score_info = _load_files_as_dict(score_file)
    tag_info = _load_files_as_dict(tag_file)

    # build nodes
    node_id_dict = dict()
    nodeid_data_dict = dict()
    edge_dict = dict()
    for k, obj in edge_info.items():
        # build new nodes
        if node_id_dict.get(obj['sentence1'], None) == None:
            new_node_id = len(node_id_dict)
            node_id_dict[obj['sentence1']] = new_node_id
            nodeid_data_dict[new_node_id] = {'sentence': obj['sentence1'], 'start': obj['start1'], 'end': obj['end1']}
        if node_id_dict.get(obj['sentence2'], None) == None:
            new_node_id = len(node_id_dict)
            node_id_dict[obj['sentence2']] = new_node_id 
            nodeid_data_dict[new_node_id] = {'sentence': obj['sentence2'], 'start': obj['start2'], 'end': obj['end2']}

        # add new edge
        _edge = (node_id_dict[obj['sentence1']], node_id_dict[obj['sentence2']])
        _edge_info = {'id': k, 'score1': list(map(float, score_info[k]['score']))[0], 'score2': list(map(float, score_info[k]['score']))[1], 'tag': tag_info[k]['tag']}

        edge_dict[_edge] = _edge_info

    return nodeid_data_dict, edge_dict

def _load_files_as_dict(path: str) -> dict:
    with open(path, 'r') as file:
        json_loaded = json.load(file)
    file.close()
    return {obj['id']: obj for obj in json_loaded}
