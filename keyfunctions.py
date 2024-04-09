import pymdp
from typing import List
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
import math
import networkx
import typing
import copy
import tempfile
import subprocess
import shutil
import os
import scipy.special

def softmax(x):
    return((np.exp(x)/np.sum(np.exp(x))))

################################### SETTING THE ENVIRONMENT / GENERATIVE PROCESS ###################################

# CREATING THE GRAPH/ENVIRONMENT FOR THE WALLETFINDING-ONLY AND EMOTIONAL-WALLETFINDING TASK - OZAN FUNCTION
def generate_connected_clusters( # 'cluster_size' refers to how many rooms you want in each chunk, and 'connections' is how many chunks of rooms you want
    cluster_size = 5, connections = 4
) -> typing.Tuple[networkx.Graph, dict]:
    edges = []
    connecting_node = 0
    while connecting_node < connections * cluster_size:
        edges += [
            (connecting_node, a)
            for a in range(
                connecting_node + 1, connecting_node + cluster_size + 1
            )
        ]
        connecting_node = len(edges)
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    return graph, {
        "locations": [
            f"hallway {i}"
            if len(list(graph.neighbors(loc))) > 1
            else f"room {i}"
            for i, loc in enumerate(graph.nodes)
        ]
    } # returns a dictionary of locations we have (location 0, location 1, etc) as rooms and hallway indices (hallway 0, room 1, room 2, etc)

# CREATING THE GRAPH/ENVIRONMENT FOR THE WALLETFINDING-ONLY AND EMOTIONAL-WALLETFINDING TASK - ADAPTED OZAN FUNCTION
def house(cluster_size = 5, connections = 4) -> typing.Tuple[networkx.Graph, dict]:
    edges = []
    connecting_node = 0
    hallway_names = ["hall", "kitchen", "bedroom", "bathroom", "gameroom", "dining room", "library"]  # add more names if needed
    hallway_counter = 0 
    while connecting_node < connections * cluster_size:
        edges += [
            (connecting_node, a)
            for a in range(
                connecting_node + 1, connecting_node + cluster_size + 1
            )
        ]
        connecting_node = len(edges)
    graph = networkx.Graph()
    graph.add_edges_from(edges)
    locations = []
    for i, loc in enumerate(graph.nodes):
        if len(list(graph.neighbors(loc))) > 1:
            location = f"{hallway_names[hallway_counter]} ({i})"
            hallway_counter += 1
        else: 
            location = f"drawer {i}"
        locations.append(location)
    return graph, {"locations": locations} 
