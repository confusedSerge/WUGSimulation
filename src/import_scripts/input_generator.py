import os
import json

def input_json_generator(word: str, id_prefix: str, pos: str, path_file: str, path_out: str):
    wu_pairs = _gen_wu_pairs_from_file(path_file)

    dict_objects = [_gen_nicolay_dict_object('{}.{}'.format(id_prefix, i), word, pos, wu1, wu2) for i, (wu1, wu2) in enumerate(wu_pairs)]

    with open(path_out, 'w') as file:
        file.write(json.dumps(dict_objects, indent=4, ensure_ascii=False))
    file.close()

def _gen_wu_pairs_from_file(path_file: str):
    wu = []
    
    with open(path_file, 'r') as file:
        line = file.readline()
        while line:
            wu.append(line)
            line = file.readline()
    file.close()

    for i in range(len(wu)):
        for j in range(i + 1, len(wu)):
            yield wu[i].strip(), wu[j].strip()

def _gen_nicolay_dict_object(id: str, lemma: str, pos: str, sentence1: str, sentence2: str):
    start1 = sentence1.lower().find(lemma)
    end1 = start1 + len(lemma)
    start2 = sentence2.lower().find(lemma)
    end2 = start2 + len(lemma)
    
    return {"id": id, "lemma": lemma, "pos": pos, "sentence1": sentence1, "sentence2": sentence2, "start1": str(start1), "end1": str(end1), "start2": str(start2), "end2": str(end2)}