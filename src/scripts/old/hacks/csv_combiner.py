import os
import pickle
import numpy as np

from csv import writer
from csv import reader

out = 'data/graphs/kw29/bigdata/metric/combined.csv'
with open(out, 'w+') as file_out:
    w = writer(file_out)
    file_to_rs = 'data/graphs/kw29/bigdata/metric/randomsampling.csv'
    with open(file_to_rs, 'r', newline='') as file:
        r = reader(file, delimiter=',')
        for i, row in enumerate(r):
            w.writerow(row)
    file.close()

    file_to_rw = 'data/graphs/kw29/bigdata/metric/randomwalk.csv'
    with open(file_to_rw, 'r', newline='') as file:
        r = reader(file, delimiter=',')
        for i, row in enumerate(r):
            if i > 0:
                w.writerow(row)
    file.close()
    
    file_to_pr = 'data/graphs/kw29/bigdata/metric/pagerank.csv'
    with open(file_to_rw, 'r', newline='') as file:
        r = reader(file, delimiter=',')
        for i, row in enumerate(r):
            if i > 0:
                w.writerow(row)
    file.close()
file_out.close()