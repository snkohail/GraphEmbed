# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Node Embedding Based on Matrix Factorization Algorithm
"""

import networkx as nx
import scipy as sp
import numpy as np
import matplotlib as plt
import tensorflow as tf

"""
Convert Edge List to Adjacency Matrix
"""

def create_adj(graph):
    