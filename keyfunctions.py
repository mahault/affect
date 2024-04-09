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
# DEFINING THE GENERATIVE PROCESS FOR THE WALLETFINDING-ONLY TASK
class wallet_Environment:
    def __init__(self, graph, init_state, object_location, location_labels): # initialising with the following variables
        self.graph = graph # setting the environment as the graph
        self.init_state = init_state # the start state
        self.state = init_state # setting the start state as the current state
        self.object_location = object_location # the true state (location) the object is in - this is what the agent infers
        self.location_labels = location_labels # the location labels returned from the 'generate_connected_clusters' function

    def move(self, action): # setting the instructions on what the consequences of the actions are in the environment (which are usually fed as observations to the agent in the next time step)
        target = int(action[0]) # the target location will be the action that's chosen (e.g., move to location 4)
        if self.graph.has_edge(self.state, target) or self.graph.has_edge(target, self.state): # if the graph has the edge that connects the target location to the location the agent is currently in, then move there
            self.state = target

        return [self.state, 0 if self.state == self.object_location else 1, 0] # return the current state the agent is in, and whether the agent is in the location where the object (wallet) is in (0 if yes, 1 if no), and the breathing rate observation (br_obs)

    def reset(self): # what to do when resetting the environment
        self.state = self.init_state # go back to start state
        return [self.state, 0 if self.init_state == self.object_location else 1, 0] # return the current state the agent is in, and whether the agent is in the location where the object (wallet) is in (0 if yes, 1 if no), and the breathing rate observation (br_obs) which is low (0)
