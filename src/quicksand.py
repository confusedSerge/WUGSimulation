from import_scripts.input_from_norm import input_json_generator

path_in = 'data/test/abbauen.csv'
path_out = 'data/test/dev.abbauen.input'

input_json_generator('abbauen', 'abbauen', 'VERB', path_in, path_out)
