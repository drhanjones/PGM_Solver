from BayesNet import BayesNet
import random
import itertools
import pandas as pd
import numpy as np
import time
import networkx as nx
from BNReasoner import BNReasoner

def choose_number_of_connections(node):

    """
    Parameters
    ----------
    node : the node we need to make connections for.

    Returns
    -------
    num_connections : a random number of connections we want to make for each node

    """
    #print('Choosing number of connections for node number', node)
    # the first node is the tree starting so can't be directed to anything yet
    if node == 1:
        num_connections = 0
        #print('not connecting any yet because this is the tree starting')
    # other nodes can be directed
    # we could do
    # num_connections = random.randint(1, node-1)
    # but the graph becomes too large so
    # limit max connections size to 5
    # if we already have more than 4 nodes
    elif node - 1 > 4:
        num_connections = random.randint(1, 5)
    else:
        num_connections = random.randint(1, node - 1)
    #print('decided on ', num_connections, 'source nodes for node', node)
    return num_connections


def make_truth_table(node, directed_connections, num_connections):
    """


    Parameters
    ----------
    node : the node which we need to make our truth table for.
    directed_connections : the connections the node has.
    num_connections : how many connections the node has.

    Returns
    -------
    asdf_truthtable : the truth table as a dataframe (as required by BayesNet).

    """
    # how many times each value can be repeated, eg False False False is 3 repeats
    complete_truthtable = []
    initialize_truthtable = list(itertools.product([True, False], repeat=num_connections + 1))

    # t=0
    # other_t = 1-t
    for statement in initialize_truthtable:
        # p = copy.copy(t)
        # other_p = 1-p
        if statement[-1] == True:
            t = random.uniform(0, 1)
            # print('p is', t)
            # print('IN HERE')
            complete_truthtable.append(list(statement) + [t])
        if statement[-1] == False:
            other_t = 1 - t
            # print('other p is', other_t)
            complete_truthtable.append(list(statement) + [other_t])

    # create header for dataframe
    header = []
    for nodename in directed_connections:
        header.append(nodename)
    header.append(str(node))
    header.append('p')

    asdf_truthtable = pd.DataFrame(data=complete_truthtable,
                                   columns=header)

    #print('the truth table \n', asdf_truthtable)

    return asdf_truthtable


def choose_nodes_to_connect(nodes_so_far, num_connections):
    """


    Parameters
    ----------
    nodes_so_far : list of the nodes that we have added to the graph so far.
    num_connections : how many connections we should make.

    Returns
    -------
    a list of the connections we want to make between nodes.

    """
    return list(np.random.choice(nodes_so_far, size=num_connections, replace=False))


def makeBayes(initial_BayesNet, num_nodes):
    # initialize everything
    dictoftruthtables = {}
    nodes_so_far = []
    connections = []

    # for each node (iterate from 1 to last node since its intuitive)
    for node in range(1, num_nodes + 1):
        #print('Working on node', node)
        num_connections = choose_number_of_connections(node)

        # randomly choose which nodes to connect
        directed_connections = choose_nodes_to_connect(nodes_so_far, num_connections)
        #print('Chosen nodes to connect', directed_connections)

        # make a truth table for these dependencies
        truth_table = make_truth_table(str(node), directed_connections, len(directed_connections))

        # add the connections to our list of total connections
        for eachnode in directed_connections:
            connections.append([eachnode] + [str(node)])
        #print('created connections', connections)

        # put our truth table into a dict as required by BayesNet
        dictoftruthtables[str(node)] = truth_table
        # add our node into the list of nodes we have so far
        nodes_so_far.append(str(node))

    # we have all our nodes at the end of the loop
    final_nodes = nodes_so_far
    # input our generated nodes, connections and truth tables into the BayesNet creator
    # connections need to be in tuple format as required by BayesNet
    initial_BayesNet.create_bn(final_nodes, list(map(tuple, connections)), dictoftruthtables)
    #print('\n Enjoy this BayesNet I made for you :) \n -------------------- \n ')
    #initial_BayesNet.draw_structure()
    return initial_BayesNet


################ Create random BayesNet ###############


node_range = range(10,26)
ordering_heuristics_list = ['min_degree', 'min_fill','random']

results_list = []
for node_size in node_range:

    initial_BayesNet = BayesNet()
    finished_BayesNet = makeBayes(initial_BayesNet, node_size)
    while True:
        if nx.is_directed_acyclic_graph(finished_BayesNet.structure):
            break
        else:
            initial_BayesNet = BayesNet()
            finished_BayesNet = makeBayes(initial_BayesNet, node_size)

    for ordering_heuristic in ordering_heuristics_list:
        for evidence_num in range(0, 5):
            if evidence_num == 0:
                print("j1")
                evidence_size = 1
            elif evidence_num == 4:
                print("j2")
                evidence_size = node_size-1
            else:
                print("j3")
                evidence_size = round(evidence_num*node_size/4)
            dp_type_list = ["Lowest Evidence", "25% Evidence", "50% Evidence", "75% Evidence", "Maximum Evidence"]
            print('Number of nodes:', node_size, 'Number of evidence nodes:', evidence_size, "evidence_num", evidence_num,'Ordering heuristic:',
                  ordering_heuristic)
            for i in range(10):
                evidence_nodes = list(np.random.choice(node_size, size=evidence_size, replace=False))
                evidence_values = np.random.choice([True, False], size=evidence_size, replace=True)
                evidence_dict = dict(zip( list(map(str,evidence_nodes)), evidence_values))
                random_reasoner = BNReasoner(finished_BayesNet)
                start = time.time()
                mpe_reasons = random_reasoner.most_probable_explanation(evidence_dict, ordering_heuristic)
                end = time.time()
                time_taken = end - start

                results_list.append({
                    "node_size" : node_size,
                    "ordering_heuristic":ordering_heuristic,
                    "evidence_size":evidence_size,
                    "evidence_type": dp_type_list[evidence_num],
                    "iterations":i,
                    "time_taken":time_taken
                })


results_df = pd.DataFrame(results_list)