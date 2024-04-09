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

# DEFINING THE GENERATIVE PROCESS FOR THE EMOTIONAL-WALLETAGENT TASK
class Environment:
    def __init__(self, graph, init_state, object_location, location_labels): # initialising with the following variables
        self.graph = graph # setting the environment as the graph
        self.init_state = init_state # the start state
        self.state = init_state # setting the start state as the current state
        self.object_location = object_location # the true state (location) the object is in - this is what the agent infers
        self.location_labels = location_labels # the location labels returned from the 'generate_connected_clusters' function

    def move(self, action, h_sur_idx): # setting the instructions on what the consequences of the actions are in the environment (which are usually fed as observations to the agent in the next time step)
        target = int(action[0]) # the target location will be the action that's chosen (e.g., move to location 4)
        if self.graph.has_edge(self.state, target) or self.graph.has_edge(target, self.state): # if the graph has the edge that connects the target location to the location the agent is currently in, then move there
            self.state = target
        
        h_sur_idx = h_sur_idx # the surprise index from the hier level
        if h_sur_idx == 1: 
            br_obs = 0 # if the surprise level is normal, the breathing rate (br_obs) is observed as being low
        else: 
            br_obs = 1 # otherwise, the breathing rate (br_obs) is observed as being high
        return [self.state, 0 if self.state == self.object_location else 1, br_obs] # return the current state the agent is in, and whether the agent is in the location where the object (wallet) is in (0 if yes, 1 if no), and the breathing rate observation (br_obs)

    def reset(self): # what to do when resetting the environment
        self.state = self.init_state # go back to start state
        return [self.state, 0 if self.init_state == self.object_location else 1, 0] # return the current state the agent is in, and whether the agent is in the location where the object (wallet) is in (0 if yes, 1 if no), and the breathing rate observation (br_obs) which is low (0)

# FUNCTION TO TEST WHETHER LISTS WITHIN A LIST ARE EQUAL TO ONE ANOTHER TO CREATE BRV
def are_sublists_equal(list_of_lists):
    if len(list_of_lists) <= 1:
        return True

    first_sublist = list_of_lists[0]
    for sublist in list_of_lists[1:]:
        if sublist != first_sublist:
            return False

    return True

################################### BUILDING THE AGENTS (WALLET FINDING AND EMOTIONAL) ###################################

# CREATING B TENSORS FOR THE WALLET FINDING AGENT BASED ON THE GRAPH/ENVIRONMENT
def generate_transitions_from_graph(graph):
    locations = list(graph.nodes()) # list of locations from the graph/environment
    moves = locations[:] # the locations are the moves the agent can make
    B = pymdp.utils.obj_array(2) # make two B tensors, one for each state factor
    B[0] = np.zeros((len(locations), len(locations), len(moves))) # make the first B tensor with the shape (number of locations (rows), number of locations (columns), number of moves the agent can conduct (slices)), and fill it with zeros
    for _from, _to, _action in itertools.product( # iterate through each location agent will be travelling from and to and actions/moves according to the tensor shape
        range(len(locations)), range(len(locations)), range(len(moves))
    ):
        if _action == _to: # when the index of action/move taken is the same as the index for the location travelling towards, 
            if graph.has_edge(_from, _to) == True: # and there is an edge between the location the agent is travelling from and towards,
                B[0][_to, _from, _action] = 1.0 # then make the element indexed [_to (row), _from (column), _action (slice)] in the tensor = 1.0
            else:
                B[0][_from, _from, _action] = 1.0 # otherwise, make the element indexed [_from (row), _from (column), _action (slice)] in the tensor = 1.0   
        
    B[0] = pymdp.utils.norm_dist(B[0]) # normally distribute the first B tensor
    
    B[1] = np.zeros((len(locations), len(locations), 1)) # make the second B tensor with the shape (number of locations (rows), number of locations (columns), one slice - this ensures this is not a controllable/action-based factor i.e. the agent cannot move the wallet (slices)) 
    B[1][:,:,0] = np.eye(len(locations)) # the wallet location stays the same
    # B[1] += 1/3 # make the beliefs about object transition volatile
    B[1] = pymdp.utils.norm_dist(B[1]) # normally distribute the second B tensor
    return B

# BUILDING THE ACTUAL WALLET FINDING AGENT AND ITS GENERATIVE MODEL
def build_agent_from_graph(graph, metadata, impreciseA, obj_loc_priors):
    self_location = metadata["locations"] # the first state factor, location of the self
    wallet_location = metadata["locations"] # the second state factor, the location of the object/wallet
    num_states = [len(self_location), len(wallet_location)]  # the number of states we have in each state factor
    num_factors = len(num_states) # the number of state factors we have
    
    self_location_o = metadata["locations"]  # the first observation modality, location of the self
    detect_wallet = metadata["detect_wallet"] # the second observation modality, whether the wallet has been detected as 'present' or 'absent'
    br_level = metadata["BR_level"] # the third observation modality, the breathing rate level
    num_observations = [len(self_location_o), len(detect_wallet), len(br_level)] # the number of observations we have in each observation modality
    num_modalities = len(num_observations) # the number of observation modalities we have
    
    # create likelihood (A) tensors
    A = pymdp.utils.obj_array(num_modalities) # create three A tensors, one for each observation modality
    
    A[0] = np.zeros((num_observations[0], num_states[0], num_states[1])) # make the first A tensor with the shape (number of locations as observed (rows), number of self_locations (columns), number of wallet_locations (slices))
    for idx in range(num_states[0]):
        A[0][idx, idx,:] = 1.0 # identity matrix for self_location - i.e., regardless of where the wallet is, the agent will observe the location of the self as where the agent is in the current timestep
    
    A[1] = np.ones((num_observations[1], num_states[0], num_states[1])) # make the second A tensor with the shape (wallet present or absent as observed (rows), number of self_locations (columns), number of wallet_locations (slices))
    A[1][0] = 1-impreciseA
    A[1][1] = impreciseA
    for loc in range(num_states[0]): # A_detect_wallet tensor where agent can only see the wallet (present) if the agent is at the location of the wallet
        A[1][0, loc, loc] = impreciseA
        A[1][1, loc, loc] = 1-impreciseA
        
    A[1] = pymdp.utils.norm_dist(A[1]) # normally distributing the second A tensor

    A[2] = np.ones((num_observations[2], num_states[0], num_states[1])) # make the third A tensor with the shape (BR level (rows), number of self_locations (columns), number of wallet_locations (slices)), and fill with ones 
    A[2] = pymdp.utils.norm_dist(A[2]) # normally distributing the third A tensor
    
    # create transition (B) tensors from graph
    B = generate_transitions_from_graph(graph) # calling the function to create B tensors that we defined above
    
    # create preference (C) tensors
    C = pymdp.utils.obj_array_zeros([num_observations[0], num_observations[1], num_observations[2]]) # make three C tensors of lengths according to the observation modalities, and fill them with zeros
    C[0] = np.zeros(num_states[0])
    C[1][0] = 1.0 # set the preference to see the wallet (present) as 1.0
    
    # create prior distribution (D) tensors 
    D = pymdp.utils.obj_array(num_factors) # creating two D tensors, one for each state factor
    D[0] = np.zeros(num_states[0])
    D[0][0] = 1.0
    
    D[1] = np.ones(num_states[0])
    for p in range(len(obj_loc_priors)):
        D[1][obj_loc_priors[p]] = 5.0
    D[1] = softmax(D[1]) # softmax the second D tensor
    D[1] = pymdp.utils.norm_dist(D[1]) # normally distribute the second D tensor
    
    return pymdp.agent.Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        action_selection="stochastic",
        policy_len=4
    ) # return the variables set up and put into the Agent class in pymdp

# POLICY SEARCH FOR PRUNING
def findpaths(g, u, n):
    if n == 0:
        return [[u]]
    return [
        [u] + path
        for neighbor in g.neighbors(u)
        for path in findpaths(g, neighbor, n - 1)
        ]

def possible_policies(s, l): 
    graph, meta = generate_connected_clusters(5,4)
    return [p[1:] for p in findpaths(graph, s, l)]

def amend_policies(policies):
    res = []
    for policy in policies:
        edited = []
        for step in policy:
            edited.append(np.array([step, 0]))
        res.append(edited)
    return res

# BUILDING THE EMOTIONAL AGENT AND ITS GENERATIVE MODEL
def build_emotional_agent():
    
    states = {"emotions": ["neutral", "anxious", "happy"]} # list out the emotions we want, add additional state factors if need be
    num_states = [len(states[state_idx]) for state_idx in states] # calculates number of states in each state factor
    num_factors = len(num_states) # calculates number of state factors

    choice_emotions = ["no_choice"]
    num_controls = [len(choice_emotions)]

    observations = {"surprise_level": ["low_surprise", "normal_surprise", "high_surprise"],
                    "BRV": ["normal", "high"]} # list out the observations and modalities we want, add additional observation modalities if need be
    num_observations = [len(observations[obs_idx]) for obs_idx in observations] # calculate number of observations in each observational modality
    num_modalities = len(num_observations) # calculate number of modalities

    # create likelihood (A) tensors
    A = pymdp.utils.obj_array(num_modalities) # creating two A tensors, one for each observation modality
    
    #                 neu  anx  hap _emotions
    A[0] = np.array([[0.0, 0.0, 1.0], # low surprise
                     [1.0, 0.0, 0.0], # normal surprise
                     [0.0, 1.0, 0.0]]) # high surprise

    #                 neu  anx  hap _emotions
    A[1] = np.array([[0.9, 0.1, 0.1], # normal BRV
                     [0.1, 0.9, 0.9]]) # high BRV
    
    A = pymdp.utils.norm_dist_obj_arr(A) # normally distribute the A tensor
    
    # create transition (B) tensors 
    B = pymdp.utils.obj_array(num_factors) # creating one B tensor for the emotion state factor
    B[0] = np.zeros( (num_states[0], num_states[0], len(choice_emotions)) ) # make the B tensor with the shape (number of emotions (rows), number of emotions (columns),  number of controllable actions (which is none here so it's of length 1; slices))
    B[0] += 1/3 # adding a small probability of transitioning to another emotion 

    B = pymdp.utils.norm_dist_obj_arr(B) # normally distribute the B tensor
    
    # create preference (C) tensors
    C = pymdp.utils.obj_array(num_modalities)
    C[0] = np.array([0.0, 0.0, 0.0])
    C[1] = np.array([0.0, 0.0])
    
    # create prior distribution tensors 
    D = pymdp.utils.obj_array(num_factors) # creating one D tensor for the emotion state factor
    for factor_idx in range(num_factors):
        D[factor_idx] = np.ones(num_states[factor_idx]) / num_states[factor_idx] # flat priors over emotion states
    
    
    # policies = pymdp.control.construct_policies(num_states, num_controls) # constructing policies
    
    return pymdp.agent.Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        action_selection="stochastic",
        policy_len=1

    ) # return the variables set up and put into the Agent class in pymdp

# GENERATE C TENSORS FOR LOWER LEVEL BASED ON EMOTIONAL STATE INFERRED AT HIER LEVEL
def generate_preference(emotion_idx, metadata):
    self_location_o = metadata["locations"]  # the first observation modality, location of the self
    detect_wallet = metadata["detect_wallet"] # the second observation modality, whether the wallet has been detected as 'present' or 'absent'
    br_level = metadata["BR_level"] # the third observation modality, the breathing rate level
    num_observations = [len(self_location_o), len(detect_wallet), len(br_level)] # the number of observations we have in each observation modality
    num_modalities = len(num_observations) # the number of observation modalities we have
    

    C = pymdp.utils.obj_array(3) # creating three C tensors, one for each observation modality
    if emotion_idx == 0: # neutral
        C[0] = np.zeros(num_observations[0])
        # C[0] = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        C[1] = np.array([1.0, 0.0]) # prefer to observe the wallet
        C[2] = np.array([0.0, 0.0]) # flat and no preferences for BR level
        
    if emotion_idx == 1: # anxious
        C[0] = np.zeros(num_observations[0])
        # C[0] = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # flat preferences to be in all locations
        # C[0] = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # flat preferences to be in all locations
        C[1] = np.array([10.0, 0.0]) # highly prefer to observe the wallet
        C[2] = np.array([0.0, 0.0]) # flat and no preferences for BR level
        
    if emotion_idx == 2: # happy
        C[0] = np.zeros(num_observations[0])
        # C[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # flat and no preferences to be in all locations
        C[1] = np.array([1.0, 0.0]) # prefer to observe the wallet
        C[2] = np.array([0.0, 0.0]) # flat and no preferences for BR level
    
    return C          

# RUNNING THE EMOTIONAL-WALLETFINDING TASK
def run_hier_model(env, lower_agent, hier_agent, lower_t, hier_t, H_threshold, metadata) -> dict:
    log = { # keeping a record of what's happening at each timestep
        "lower_observations": [],
        "lower_obs2" : [],
        "lower_beliefs": [],
        "lower_q_pi": [],
        "lower_efe": [],
        "lower_actions": [],
        "lower_q_entropy": [],
        "hier_observations": [],
        "hier_beliefs": []
    } 
    lower_agent.reset() # reset the lower agent
    l_obs = env.reset() # reset the environment
    init_h_obs = [1,0] # initial observations at hier level are normal surprise level and normal BRV

    log["lower_observations"].append((l_obs,)) # recording lower agent's observations
    log["lower_obs2"].append((l_obs)) # recording lower agent's observations in a different format
    
    l_qs = lower_agent.infer_states(l_obs) # lower agent infers states based on observations
    log["lower_beliefs"].append((l_qs,)) # recording lower agent's beliefs

    H = 0.0 # posterior entropy 
    ln_qs = pymdp.maths.spm_log_obj_array(l_qs[1]) # obtaining record of the lower agent's posterior beliefs 
    H -= l_qs[1].dot(ln_qs) # calculating posterior entropy of lower agent
    log["lower_q_entropy"].append((H)) # recording lower agent's posterior entropy

    h_sur_idx = 1 # normal surprise level indexed
    log["lower_actions"].append(None) # no actions conducted at current timestep
    log["hier_observations"].append(init_h_obs) # recording hier agent's initial observations
    
    h_qs = hier_agent.infer_states(init_h_obs) # hier agent infers states based on observations
    log["hier_beliefs"].append(h_qs) # recording hier agent's beliefs

    states = {"emotions": ["neutral", "anxious", "happy"]}  # list out the emotions we want, add additional state factors if need be
    num_states = [len(states[state_idx]) for state_idx in states] # calculates number of states in each state factor
    # num_factors = len(num_states)  # calculates number of state factors

    choice_emotions = ["no_choice"]
    # num_controls = [len(choice_emotions)]

    observations = {"surprise_level": ["low_surprise", "normal_surprise", "high_surprise"],
                    "BRV": ["normal", "high"]} # list out the observations and modalities we want, add additional observation modalities if need be
    num_observations = [len(observations[obs_idx]) for obs_idx in observations] # calculates the number of observations in each modality
    # num_modalities = len(num_observations) # calculates number of modalities
    
    # state = [0]

    for ht in range(hier_t):
        # h_emo_idx = np.argmax(h_qs)
        for lt in range(lower_t):
            count = lt + 1
            loweragent_possiblepolicies = possible_policies(np.argmax(l_qs[0]), 5)
            lower_agent.policies = np.array(amend_policies(loweragent_possiblepolicies))
            lower_agent.E = np.ones(len(loweragent_possiblepolicies)) / len(loweragent_possiblepolicies)
            
            q_pi, efe = lower_agent.infer_policies() # lower agent infers policies
            log["lower_q_pi"].append((q_pi,)) # recording the action
            log["lower_efe"].append((efe,)) # recording the action
            lower_action = lower_agent.sample_action() # lower agent samples actions
            log["lower_actions"].append((lower_action,)) # recording the action
            
            l_obs = env.move(lower_action, h_sur_idx) # producing observations for lower agent via the generative process/environment, which takes the surprise index from hier level and the action the lower agent executed
            log["lower_observations"].append((l_obs,)) # recording the lower agent's observations
            log["lower_obs2"].append((l_obs)) # recording the lower agent's observations in a different format
            
            l_qs = lower_agent.infer_states(l_obs) # lower agent infers states based on observations
            log["lower_beliefs"].append((l_qs,)) # recording the lower agent's beliefs
            
            H = 0. # posterior entropy 
            ln_qs = pymdp.maths.spm_log_obj_array(l_qs[1]) # obtaining record of the lower agent's posterior beliefs 
            H -= l_qs[1].dot(ln_qs) # calculating posterior entropy of lower agent
            log["lower_q_entropy"].append((H)) # recording lower agent's posterior entropy
            # print(H)
            
            # print('\033[1m' +"lower_timestep: " + str(count) + '\033[0m')
            # print()
            # print(f"lower_Actions: {lower_action}")
            # print()
            # print(f"lower_Observations: {l_obs}")
            # print()
            # print(f"lower_Posterior beliefs: {l_qs}")
            # print()
            # print(f"lower_Entropy: {H}")
            # print()
            # print(lower_agent.policies)
            # print()
            # print()

            h_obs = [] # empty hier level observations 
            if H > H_threshold: # if the posterior entropy is higher than 1,
                h_obs.append(2) # append '2' (high surprise) to the hier level observations
                h_sur_idx = 2 # the surprise level index is 2
            elif 0.0 < H < 0.1: # if the posterior entropy is between 0 and 0.1,
                h_obs.append(0) # append '0' (low surprise) to the hier level observations
                h_sur_idx = 0 # the surprise level index is 0
            else: # if the posterior entropy is anything else (basically between 0.1 and 1),
                h_obs.append(1) # append '1' (normal surprise) to the hier level observations
                h_sur_idx = 1 # the surprise level index is 1

            # if H > 1.0: 
            #     h_obs = [2,1]
            # elif 0.0 < H < 0.1:
            #     h_obs = [0,0]
            # else:
            #     h_obs = [1,1]
            
            brv_ref = [sub_list[-2:] for sub_list in log["lower_obs2"][-2:]] # creating a reference list for the BRV using the lower agent's BR observations from the previous two timesteps
            brv_ref_2 = [sub_list[-1:] for sub_list in brv_ref] # comparing the values within the reference list
            
            if are_sublists_equal(brv_ref_2) == True: # if the values in the BRV reference list are equal,
                h_obs.append(0) # append '0' (normal BRV) to the hier level observations 
            else: # if the values in the BRV reference list are varied
                h_obs.append(1) # append '1' (high BRV) to the hier level observations

            log["hier_observations"].append(h_obs) # record the hier level observations

            
        h_qs = hier_agent.infer_states(h_obs) # hier agent infers states based on observations
        log["hier_beliefs"].append(h_qs) # recording the hier agent's beliefs

        emotion_idx = np.argmax(h_qs[0], axis=None) # getting the index for the emotional state inferred
        lower_agent_C = generate_preference(emotion_idx, metadata) # creating C tensors for lower agent depending on emotional state inferred
        lower_agent.C = lower_agent_C # setting the C tensors as the C tensor for the lower agent

        # print('\033[1m' + f"\nhier_timestep {ht} :" + '\033[0m')
        # print()
        # # plot_posterior_entropy(H, "posterior entropy")
        # print(H)
        # print(l_qs[1])
        # print(h_qs[0])
        # print(np.argmax(h_qs[0], axis=None))
        # print()
        # print(f"hier_prior belief of state: \n{[(list(states)[f], states[list(states)[f]][state[f]]) for f in range(len(states))]}")
        # print()
        # print(f"hier_observations: \n{[(list(observations)[g], observations[list(observations)[g]][h_obs[g]]) for g in range(len(observations))]}")
        # print()
        # print(f"hier_posterior beliefs: " + str(h_qs))
        # print()
        # print()

        if l_obs[1] == 0: # if the wallet is found, break the loop
            break
        # # plotting beliefs
        # plot_emo_beliefs(hier_agent.qs[0], "hier_posterior belief of emotion at time {}".format(ht))
    return log

################################### VISUALISATIONS ###################################

# TO VISUALISE THE STATES ON THE GRAPH
def visualize_state(graph, agent_locs, item_loc, meta, ax=None):
    color_map = ["lightgrey"] * len(list(graph.nodes()))
    _colors = copy.deepcopy(matplotlib.colors.TABLEAU_COLORS)
    # del _colors["b"]
    del _colors["tab:red"]
    _colors = list(_colors.keys())
    color_map[item_loc] = "tab:red"
    for i, agent in enumerate(agent_locs):
        color_map[agent] = _colors[i]
    random_pos = networkx.random_layout(graph, seed=42)
    pos = networkx.spring_layout(graph, pos=random_pos, k=0.3)

    node_shapes = {}
    for node in graph.nodes():
        if node == item_loc:
            node_shapes[node] = "s"  
        else:
            node_shapes[node] = "o"  
    
    for node, shape in node_shapes.items():
        networkx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[node], node_color=color_map[node], node_shape=shape, node_size=700)
    
    labels = {node: meta["locations"][node] for node in graph.nodes()}  # create labels dictionary
    networkx.draw_networkx_labels(graph, pos, labels = labels, ax=ax, font_color="black", font_size = 20)
    networkx.draw_networkx_edges(graph, pos, ax=ax)
    # plt.show()
    # networkx.draw(graph, ax=ax, node_color=color_map, with_labels=True, pos=pos)

# FUNCTIONS TO CHANGE ACTIONS AND OBSERVATIONS TO STRINGS
def obs_to_string(obs, outcome_labels, obj_outcome_labels, br_labels):
    return f"Observation (lower level): {outcome_labels[obs[0]]} & {obj_outcome_labels[obs[1]]} & {br_labels[obs[2]]}"

def h_obs_to_string(obs, surprise_labels, BRV_labels):
    return f"Observation (hier level): {surprise_labels[obs[0]]} & {BRV_labels[obs[1]]}"

def br_to_string(obs, br_labels):
    return f"BR: {br_labels[obs[2]]}"

def brv_to_string(obs, BRV_labels):
    return f"BRV: {BRV_labels[obs[1]]}"


def action_to_str(action, move_labels):
    return f"Action (lower level): {move_labels[action[0]]}"

# REPORTING THE RECORD FOR WALLETFINDING-ONLY TASK
def report_replay(graph, replay_log, item_location, meta):
    lower_observations = replay_log["lower_observations"]
    num_agents = len(replay_log["lower_observations"][0])
    lower_beliefs = replay_log["lower_beliefs"]
    lower_actions = replay_log["lower_actions"]
    hier_observations = replay_log["hier_observations"]
    hier_beliefs = replay_log["hier_beliefs"]

    fig = plt.figure(constrained_layout=False, figsize=(40, 20))
    widths = [1, 1, 2.5]
    spec = fig.add_gridspec(nrows=2, ncols=3, width_ratios=widths)
    axes = []
    t = 0
    num_zeros = int(math.log10(len(lower_observations))) + 1
    
    axes.append(fig.add_subplot(spec[0, 0]))
    axes.append(fig.add_subplot(spec[0, 1]))
    axes.append(fig.add_subplot(spec[1, 0]))
    axes.append(fig.add_subplot(spec[1, 1]))
    axes.append(fig.add_subplot(spec[:, 2]))
    with tempfile.TemporaryDirectory() as dir:
        for i, (l_obs, l_qs, l_act, h_obs, h_qs) in enumerate(zip(lower_observations, lower_beliefs, lower_actions, hier_observations, hier_beliefs)):
            # print()
            # print()
            # print("timestep: " + str(t))
            t += 1
            plot_idx = 0
            for agent in range(num_agents):
                if l_act:
                    act_str = action_to_str(l_act[agent].astype(int), meta["locations"])
                    # print(f"<agent {agent}> {act_str}")
                plot_beliefs(l_qs[agent], h_qs[agent], "", [axes[plot_idx], axes[plot_idx + 1], axes[plot_idx + 2]])
                l_o = np.array(l_obs)[:, 0]
                # h_o = np.array(h_obs)[:, 0]
                l_obs_str = obs_to_string(
                    l_obs[agent], meta["locations"], meta["detect_wallet"], meta["BR_level"]
                )
                h_obs_str = h_obs_to_string(
                    h_obs, meta["surprise_level"], meta["BRV_level"]
                )

                br_str = br_to_string(
                    l_obs[agent], meta["BR_level"]
                )
                brv_str = brv_to_string(
                    h_obs, meta["BRV_level"]
                )
                # print(f"<agent {agent}> {l_obs_str}")
                # print(f"<agent {agent}> {h_obs_str}")

                # add timestep and BR_level and BRV_level to a subplot
                axes[3].clear() 
                axes[3].text(0.5, 0.5, f"Timestep: {t} \n{br_str} \n{brv_str}", 
                             horizontalalignment='center', verticalalignment='center', 
                             transform=axes[3].transAxes, fontsize = 30)
                axes[3].axis('off')  # hide the axis

                plot_idx += 3
            l_o = np.array(l_obs)[:, 0]
            visualize_state(graph, l_o, item_location, meta, axes[-1])
            format_modifier = f"0{num_zeros}d"
            fig.savefig(f"{dir}/{i:{format_modifier}}.png")
            for ax in axes:
                ax.clear()
        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "1",
                "-pattern_type",
                "glob",
                "-i",
                f"{dir}/*.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "1",
                f"{dir}/out.mp4",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.copy(f"{dir}/out.mp4", f"{os.getcwd()}/out.mp4")
    return f"{os.getcwd()}/out.mp4"

def evaluate_found(log):
    amount_agents = len(log["lower_observations"][0])
    visited = []
    for agent in range(amount_agents):
        visited.extend([obs[agent][1] for obs in log["lower_observations"]])
    return any([state == 0 for state in visited])


def evaluate_length(log):
    return len(log["lower_observations"])




# RUNNING WALLET AGENT ACTIVE INFERENCE 
def wallet_agent_run(wallet_env, agent, max_steps) -> dict:
    log = {
        "lower_observations": [],
        "lower_beliefs": [],
        "lower_actions": [],
        "q_pi": [],
        "efe": [],
        "lower_q_entropy": []
    }
        
    agent.reset()
    obs = wallet_env.reset()
    log["lower_observations"].append((obs,))
    qs = agent.infer_states(obs)
    log["lower_beliefs"].append((qs,))
    log["lower_actions"].append(None)
    log["q_pi"].append(None)
    log["efe"].append(None)
    log["lower_q_entropy"].append(None)
    
    for _ in range(max_steps):

        loweragent_possiblepolicies = possible_policies(np.argmax(qs[0]), 5)
        agent.policies = np.array(amend_policies(loweragent_possiblepolicies))
        agent.E = np.ones(len(loweragent_possiblepolicies)) / len(loweragent_possiblepolicies)
        
        q_pi, efe = agent.infer_policies()
        log["q_pi"].append((q_pi,))
        log["efe"].append((efe,))
        
        action = agent.sample_action()
        log["lower_actions"].append((action,))
        
        obs = wallet_env.move(action)
        log["lower_observations"].append((obs,))
        
        qs = agent.infer_states(obs)
        log["lower_beliefs"].append((qs,))
        
        H = 0. # posterior entropy 
        ln_qs = pymdp.maths.spm_log_obj_array(qs[1])
        H -= qs[1].dot(ln_qs)
        log["lower_q_entropy"].append((H))
        

        # print("timestep: " + str(_))
        # print()
        # print(f"Actions: {action}")
        # print()
        # print(f"Observations: {obs}")
        # print()
        # print(f"Posterior beliefs: {qs}")
        # print()
        # print(f"Entropy: {H}")
        # print()
        # print()
                
        if obs[1] == 0:
            break
    return log


def report_replay_wallet(graph, replay_log, item_location, meta):
    lower_observations = replay_log["lower_observations"]
    num_agents = len(replay_log["lower_observations"][0])
    lower_beliefs = replay_log["lower_beliefs"]
    lower_actions = replay_log["lower_actions"]

    fig = plt.figure(constrained_layout=False, figsize=(40, num_agents * 10))
    widths = [1, 1, 2]
    spec = fig.add_gridspec(nrows=num_agents, ncols=3, width_ratios=widths)
    axes = []
    t = 0
    num_zeros = int(math.log10(len(lower_observations))) + 1
    for i in range(num_agents):
        axes.append(fig.add_subplot(spec[i, 0]))
        axes.append(fig.add_subplot(spec[i, 1]))
    axes.append(fig.add_subplot(spec[:, 2]))
    with tempfile.TemporaryDirectory() as dir:
        for i, (l_obs, l_qs, l_act) in enumerate(zip(lower_observations, lower_beliefs, lower_actions)):
            # print()
            # print()
            # print("timestep: " + str(t))
            t += 1
            plot_idx = 0
            for agent in range(num_agents):
                if l_act:
                    act_str = action_to_str(l_act[agent].astype(int), meta["locations"])
                    # print(f"<agent {agent}> {act_str}")
                plot_wallet_beliefs(l_qs[agent], "", [axes[plot_idx], axes[plot_idx + 1]])
                l_o = np.array(l_obs)[:, 0]
                # h_o = np.array(h_obs)[:, 0]
                l_obs_str = obs_to_string(
                    l_obs[agent], meta["locations"], meta["detect_wallet"]
                )
                
                # print(f"<agent {agent}> {l_obs_str}")

                plot_idx += 2
            l_o = np.array(l_obs)[:, 0]
            visualize_state(graph, l_o, item_location, axes[-1])
            format_modifier = f"0{num_zeros}d"
            fig.savefig(f"{dir}/{i:{format_modifier}}.png")
            for ax in axes:
                ax.clear()
        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "1",
                "-pattern_type",
                "glob",
                "-i",
                f"{dir}/*.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "1",
                f"{dir}/out.mp4",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.copy(f"{dir}/out.mp4", f"{os.getcwd()}/out.mp4")
    return f"{os.getcwd()}/out.mp4"

# BAR CHARTS FOR PLOTTING BELIEFS OF EMOTIONAL-WALLETFINDING TASK
def plot_beliefs(l_belief_dist, h_belief_dist, title_str="", axes=None):
    """
    plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """
    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))
        fig.suptitle(title_str)
    axes[0].grid(zorder=0)
    axes[0].bar(range(l_belief_dist[0].shape[0]), l_belief_dist[0], color="k", zorder=3)
    xrange = list(range(l_belief_dist[0].shape[0]))
    axes[0].set_xticks(xrange)
    axes[0].set_ylim([0, 1])
    axes[0].set_title('Beliefs about self_location', fontsize = 24)
    axes[0].tick_params(axis='y', which='major', labelsize=14)
    axes[0].tick_params(axis='x', which='major', labelsize=12)

    axes[1].grid(zorder=0)
    axes[1].bar(range(l_belief_dist[1].shape[0]), l_belief_dist[1], color="b", zorder=3)
    xrange1 = list(range(l_belief_dist[1].shape[0]))
    axes[1].set_xticks(xrange1)
    axes[1].set_ylim([0, 1])
    axes[1].set_title('Beliefs about wallet_location', fontsize = 24)
    axes[1].tick_params(axis='y', which='major', labelsize=14)
    axes[1].tick_params(axis='x', which='major', labelsize=12)

    axes[2].grid(zorder=0)
    axes[2].bar(range(h_belief_dist.shape[0]), h_belief_dist, color="r", zorder=3)
    xrange2 = list(range(h_belief_dist.shape[0]))
    axes[2].set_xticks(xrange2, labels = ("neutral", "anxious", "happy"))
    axes[2].set_ylim([0, 1])
    axes[2].set_title('Beliefs about emotional_state', fontsize = 24)
    axes[2].tick_params(axis='y', which='major', labelsize=14)
    axes[2].tick_params(axis='x', which='major', labelsize=24)

# BAR CHARTS FOR PLOTTING BELIEFS OF WALLETFINDING-ONLY TASK
def plot_wallet_beliefs(l_belief_dist, title_str="", axes=None):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        fig.suptitle(title_str)
    axes[0].grid(zorder=0)
    axes[0].bar(range(l_belief_dist[0].shape[0]), l_belief_dist[0], color="k", zorder=3)
    xrange = list(range(l_belief_dist[0].shape[0]))
    axes[0].set_xticks(xrange)

    axes[1].grid(zorder=0)
    axes[1].bar(range(l_belief_dist[1].shape[0]), l_belief_dist[1], color="b", zorder=3)
    xrange1 = list(range(l_belief_dist[1].shape[0]))
    axes[1].set_xticks(xrange1)

# PLOTTING POSTERIOR ENTROPY
def plot_posterior_entropy(Hs, title=""):
    plt.grid(zorder=0)
    plt.bar(range(Hs.shape[0]), Hs, color='r')
    plt.xticks(range(Hs.shape[0]))
    plt.title(title)
    plt.show()

# FOR EXPERIMENT LOGGING
def evaluate_coverage(log, graph):
    amount_agents = len(log["lower_observations"][0])
    visited_nodes = set()
    time_len = len(log["lower_observations"])
    coverage = np.zeros(time_len)
    for t in range(time_len):
        for agent in range(amount_agents):
            obs = log["lower_observations"][t][agent][0]
            visited_nodes.add((obs))
        coverage[t] = len(visited_nodes)
    all_nodes = list(graph.nodes())
    return coverage / len(all_nodes)

