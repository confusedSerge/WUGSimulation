import os
import json
import re

def input_json_generator(word: str, id_prefix: str, pos: str, path_file: str, path_out: str):
    wu_pairs = _gen_wu_pairs_from_file(path_file)

    dict_objects = [_gen_nicolay_dict_object('{}.{}'.format(id_prefix, i), word, pos, wu1, wu2) for i, (wu1, wu2) in enumerate(wu_pairs)]

    with open(path_out, 'w') as file:
        file.write(json.dumps(dict_objects, indent=4, ensure_ascii=False))
    file.close()

def _gen_wu_pairs_from_file(path_file: str):
    wu = []
    
    with open(path_file, 'r') as file:
        file.readline() # skipping header
        line = file.readline()
        while line:
            identifier, sentence_normalized = line.split('\t')
            position = int(identifier.split('-')[-1])
            wu.append([*_get_char_start_end_pos(sentence_normalized.strip(), position), sentence_normalized.strip()])
            line = file.readline()
    file.close()

    for i in range(len(wu)):
        for j in range(i + 1, len(wu)):
            yield wu[i], wu[j]

def _gen_nicolay_dict_object(id: str, lemma: str, pos: str, sentence1: list, sentence2: list):
    start1 = sentence1[0]
    end1 = sentence1[1]
    start2 = sentence2[0]
    end2 = sentence2[1]
    
    return {"id": id, "lemma": lemma, "pos": pos, "sentence1": sentence1[2], "sentence2": sentence2[2], "start1": str(start1), "end1": str(end1), "start2": str(start2), "end2": str(end2)}

def _get_char_start_end_pos(sentence: str, position: int):    
    word = re.findall(r"[\w^-]+|[^\s\w]", sentence, re.UNICODE)[position]
    print(sentence[:10], word)
    start = len(sentence.split(word)[0])
    end = start + len(word)
    return start, end