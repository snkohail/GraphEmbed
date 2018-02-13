# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Node Embedding Based on Matrix Factorization Algorithm
"""

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import networkx as nx
import matplotlib as plt
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
"""
Convert Edge List to Adjacency Matrix
"""
def read_edges(filename):    
    g = nx.read_edgelist(filename, nodetype=str, data=(('weight',float),),create_using=nx.DiGraph())
    return g
def build_dataset(graph):
    index = 0
    number_of_edges = graph.number_of_edges()
    dataset = np.ndarray(shape=(number_of_edges), dtype=np.int32)
    labels = np.ndarray(shape=(number_of_edges, 1), dtype=np.int32)
    dictionary = {k: v for v, k in enumerate(graph.nodes)}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for e in graph.edges:
        dataset[index] = dictionary[e[0]]
        labels[index] = dictionary[e[1]]
        index = index+1
    return dataset, labels, dictionary, reverse_dictionary
    
filename = '/Users/skohail/Desktop/Graph_Embd_Project/graph'
G = read_edges(filename)
"""print(G.edges(data=True))"""
node_size = G.number_of_nodes()
print("Number of nodes: %d " %node_size)
d,l,_,_=build_dataset(G)
print (len(d))