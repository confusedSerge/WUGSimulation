import os
from import_scripts.input_generator import input_json_generator

for path, dirnames, files in os.walk('data/garrafao/data_unreleased_please_no_share'):
    print(path)
    split_path = path.split('/')
    if len(dirnames) == 0: 
        print('Generating: Lang: {}, lemma: {}'.format(split_path[-2], split_path[-1]))
        input_json_generator('dev.{}'.format(split_path[-1]), split_path[-2], '{}/uses.csv'.format(path), '{}/dev.{}.data'.format(path, split_path[-1]))
