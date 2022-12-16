from typing import Union

import networkx

from BayesNet import BayesNet
from copy import deepcopy
import pandas as pd
import numpy as np
import random
import time
import itertools


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net


    # TODO: This is where your methods should go


    def factor_multiplication(self, f, g):
        common_columns = list(set(f.columns) & set(g.columns))
        common_columns = [i for i in common_columns if i!='p']

        if len(common_columns) == 0:
            return None
        joined_df = f.merge(g, on=common_columns, how="outer")
        #print(joined_df)
        joined_df['p'] = joined_df['p_x'] * joined_df['p_y']
        joined_df.drop(['p_x', 'p_y'], axis=1, inplace=True)
        return joined_df

    def network_prune(self, G, Q, E):

        G_copy = deepcopy(G)

        while True:

            G_c_nodes = G_copy.get_all_variables()

            t1=True
            for node in G_c_nodes:
                if node in Q or E:
                    continue
                children = G_copy.get_children(node)
                if len(children) == 0:
                    G_copy.del_var(node)
                    t1 = False

            for node in E:
                n_children = G_copy.get_children(node)
                if len(n_children) !=0:
                    [G_copy.del_edge(x,node) for x in n_children]
                    t2 = False
                else:
                    t2 = True

            if t1 and t2:
                break

        return G_copy

    def d_separation(self, X,Y,Z):

        pruned_graph = self.network_prune(self.bn,X,Z)
        pruned_graph_nodes = pruned_graph.get_all_variables()

        if len(list(set(pruned_graph_nodes) & Y)) == 0:
            return True
        else:
            return False

    def independence(self, X, Y, Z):

        if self.d_separation(X,Y,Z):
            return True
        else:
            return False

    def calulate_marginalisation(self, X, f):
        df = f.copy(deep=True)
        gby_list = [e for e in list(df.columns) if e not in ['p', X]]
        if len(gby_list) !=0:
            df_ret = df.drop(X, axis=1).groupby(by=gby_list).sum().reset_index()
            return df_ret


    def maxing_out(self, X, f):
        df = f.copy(deep=True)
        gby_list = [e for e in list(df.columns) if e not in ['p', X] and not e.startswith("inst_")]

        if len(gby_list) !=0:
            idx = df.groupby(by=gby_list)['p'].transform(max) == df['p']
            df = df[idx].reset_index(drop=True)
            df = df.groupby(by=gby_list).last().reset_index()
        else:
            df = df[df['p'] == df['p'].max()]

        df = df.rename(columns={X: "inst_"+X})
        return df


    def min_degree(self):

        ordering_set = []
        graph = self.bn.get_interaction_graph()

        while len(graph.nodes) > 0:
            min_degree_node = min(graph.nodes, key=graph.degree)
            ordering_set.append(min_degree_node)
            graph.remove_node(min_degree_node)

        print(ordering_set)
        return ordering_set

    def min_fill_graph(self,graph):

        mf_heuristic = list(graph.nodes)[0]
        f_val = 0
        for node in graph.nodes:
            f_count = 0
            node_neighbors = graph.neighbors(node)
            for n1,n2 in itertools.combinations(node_neighbors,2):
                if n1 not in graph.neighbors(n2):
                    f_count +=1
            if f_count < f_val:
                mf_heuristic = node
                f_val = f_count

        return mf_heuristic

    def min_fill(self):

        ordering_set = []
        graph = self.bn.get_interaction_graph()
        while len(graph.nodes) > 0:
            min_degree_node = self.min_fill_graph(graph)
            ordering_set.append(min_degree_node)
            graph.remove_node(min_degree_node)
        return ordering_set


    def ordering(self, ordering_heuristics = "minfill", X = None):

        if ordering_heuristics == "random":
            if X is not None:
                ordering_set = set(self.bn.get_all_variables()) - set(X)
            else:
                ordering_set = set(self.bn.get_all_variables())
            ordering_set = list(ordering_set)
            random.shuffle(ordering_set)
        elif ordering_heuristics == "min_fill":
            ordering_set = self.min_fill()
        elif ordering_heuristics == "min_degree":
            ordering_set = self.min_degree()

        return ordering_set

    def factor_multiplication_multiple(self, f_list):

            f_a = f_list.pop(0)
            while len(f_list) > 0:
                f_b = f_list.pop(0)
                f_c = self.factor_multiplication(f_a, f_b)
                if f_c is not None:
                    f_a = f_c
                else:
                    f_list.append(f_b)

            return f_a

    def variable_elimination(self, cpt_dict, ordering_set = None, elim_type = 'sumout', X = None):

        if ordering_set is None:
            ordering_set = self.ordering(ordering_heuristics= "min_degree", X = X)
        current_cpt_dict = deepcopy(cpt_dict)
        for var in ordering_set:
            sum_out_list = [x for x in current_cpt_dict.keys() if var in current_cpt_dict[x].columns]
            if elim_type == 'sumout':
                joint_df = self.factor_multiplication_multiple([current_cpt_dict[x] for x in sum_out_list])
                df_wo_var = self.calulate_marginalisation(var, joint_df)

            elif elim_type == 'maxout':
                joint_df = self.factor_multiplication_multiple([current_cpt_dict[x] for x in sum_out_list])
                df_wo_var = self.maxing_out(var, joint_df)

            for d_var in sum_out_list:
                current_cpt_dict.pop(d_var)
                if df_wo_var is not None:
                    current_cpt_dict['wo_'+var] = df_wo_var

        if elim_type == 'sumout':
            return current_cpt_dict
        elif elim_type == 'maxout':
            return current_cpt_dict


    def update_cpt_with_evidence(self, cpt, evidence):

        if evidence is None or evidence == {}:
            return cpt

        cpt_updated_evidence = deepcopy(cpt)
        for var in cpt_updated_evidence.keys():
            common_evidence = {x: evidence[x] for x in evidence.keys() if x in cpt_updated_evidence[var].columns}
            updated_table = self.bn.get_compatible_instantiations_table(pd.Series(common_evidence), cpt_updated_evidence[var])
            updated_table = updated_table.reset_index(drop= True)
            cpt_updated_evidence[var] = updated_table

        return cpt_updated_evidence

    def joint_marginal(self, Q, evidence=None):

        cpt_list = self.bn.get_all_cpts()

        if evidence is None or evidence == {}:
            evidence_cpt_list = deepcopy(cpt_list)
        else:
            evidence_cpt_list = self.update_cpt_with_evidence(cpt_list, evidence)

        total_ordering_set = self.ordering(ordering_heuristics="min_degree")
        Q_ordering_set = [x for x in total_ordering_set if x in Q]
        elim_ordering_set = [x for x in total_ordering_set if x not in Q]

        joint_marginal_cpts = self.variable_elimination(evidence_cpt_list, ordering_set=elim_ordering_set, elim_type = 'sumout')
        final_joint_df = self.factor_multiplication_multiple(joint_marginal_cpts)

        return final_joint_df

    def marginal_distribution(self, Q, e=None):

        marginals_df = self.joint_marginal(Q, evidence=e)
        marginals_df['p'] = marginals_df['p']/marginals_df['p'].sum()

        return marginals_df

    def maximum_a_posteriori(self, Q, e):

        default_cpt_list = self.bn.get_all_cpts()
        evidence_cpt_list = self.update_cpt_with_evidence(default_cpt_list, e)
        total_ordering_set = self.ordering(ordering_heuristics= "min_degree")
        Q_ordering_set = [x for x in total_ordering_set if x in Q]
        elim_ordering_set = [x for x in total_ordering_set if x not in Q]

        joint_marginal_cpts = self.variable_elimination(evidence_cpt_list, ordering_set=elim_ordering_set, elim_type = 'sumout')

        joint_marginal_cpts = self.variable_elimination(joint_marginal_cpts, ordering_set = Q_ordering_set, elim_type = 'maxout')

        print("H1 \n",joint_marginal_cpts)
        #return maxout_return, instantiation

    def most_probable_explanation(self, e):

        default_cpt_list = self.bn.get_all_cpts()
        evidence_cpt_dict = self.update_cpt_with_evidence(default_cpt_list, e)

        total_ordering_set = self.ordering(ordering_heuristics= "min_fill")

        joint_marginal_dict = evidence_cpt_dict
        for elim_var in total_ordering_set:
            joint_marginal_df = self.variable_elimination(joint_marginal_dict, ordering_set=set(elim_var), elim_type='maxout')

        if len(joint_marginal_df) > 1:
            final_factor_multiplication = self.factor_multiplication_multiple(list(joint_marginal_df.values()))
        else:
            final_factor_multiplication = list(joint_marginal_df.values())[0]
        return final_factor_multiplication[final_factor_multiplication['p'] == final_factor_multiplication['p'].max()]


    

def choose_number_of_connections(node):
    """
    

    Parameters
    ----------
    node : the node we need to make connections for.

    Returns
    -------
    num_connections : a random number of connections we want to make for each node

    """
    print('Choosing number of connections for node number', node)
    # the first node is the tree starting so can't be directed to anything yet
    if node == 1:
        num_connections = 0
        print('not connecting any yet because this is the tree starting')
    # other nodes can be directed
    # we could do
    # num_connections = random.randint(1, node-1)
    # but the graph becomes too large so
    # limit max connections size to 5
    # if we already have more than 4 nodes
    elif node-1 > 4:
        num_connections = random.randint(1, 5)
    else:
        num_connections = random.randint(1, node-1)
    print('decided on ', num_connections, 'source nodes for node', node)
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
    initialize_truthtable = list(itertools.product([True, False], repeat=num_connections+1))
    
    # t=0
    # other_t = 1-t
    for statement in initialize_truthtable:
        # p = copy.copy(t)
        # other_p = 1-p
        if statement[-1] == True:
            t = random.uniform(0,1)
            # print('p is', t)
            # print('IN HERE')
            complete_truthtable.append(list(statement)+[t])
        if statement[-1] == False:
            other_t=1-t
            # print('other p is', other_t)
            complete_truthtable.append(list(statement)+[other_t])
    
    # create header for dataframe
    header = []
    for nodename in directed_connections:
        header.append(nodename)
    header.append(str(node))
    header.append('p')
    
    asdf_truthtable = pd.DataFrame(data = complete_truthtable, 
                    columns = header)

    print('the truth table \n', asdf_truthtable)

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
    for node in range(1, num_nodes+1):
        print('Working on node', node)
        num_connections = choose_number_of_connections(node)

        # randomly choose which nodes to connect
        directed_connections = choose_nodes_to_connect(nodes_so_far, num_connections)
        print('Chosen nodes to connect', directed_connections)
        
        # make a truth table for these dependencies
        truth_table = make_truth_table(str(node), directed_connections, len(directed_connections))
        
        # add the connections to our list of total connections
        for eachnode in directed_connections:
            connections.append([eachnode]+[str(node)])
        print('created connections', connections)

        # put our truth table into a dict as required by BayesNet
        dictoftruthtables[str(node)] = truth_table
        # add our node into the list of nodes we have so far
        nodes_so_far.append(str(node))
    
    # we have all our nodes at the end of the loop
    final_nodes =  nodes_so_far   
    # input our generated nodes, connections and truth tables into the BayesNet creator
    # connections need to be in tuple format as required by BayesNet
    initial_BayesNet.create_bn(final_nodes, list(map(tuple, connections)), dictoftruthtables)
    print('\n Enjoy this BayesNet I made for you :) \n -------------------- \n ')
    initial_BayesNet.draw_structure()
    return initial_BayesNet




################ Create random BayesNet ###############

numberofnodes = 7
initial_BayesNet = BayesNet()
finished_BayesNet = makeBayes(initial_BayesNet, numberofnodes)


#dog_example_path = 'testing/dog_problem.BIFXML'
#dog_example_path = 'testing/lecture_example.BIFXML'
#dog_bn = BNReasoner(dog_example_path)


le2_path = 'testing/lecture_example2.BIFXML'
le2_bn = BNReasoner(le2_path)

x = le2_bn.bn.get_all_cpts()
#print(x)
#y = le2_bn.update_cpt_with_evidence(x,{'Winter?': True, 'Sprinkler?': False})

#print(le2_bn.bn.draw_structure())

#print("\n",dog_bn.bn.get_compatible_instantiations_table(pd.Series({'Rain?': True}), x['Rain?']))
#x1 = x['Rain?']
#print(x1)

#abc = le2_bn.marginal_distribution(['Wet Grass?',"Slippery Road?"], {'Winter?': True, 'Sprinkler?': False})
#abc = le2_bn.marginal_distribution(["J","O"])
#print(abc)
#abc = le2_bn.maximum_a_posteriori(['I',"J"], {'O': True})

print(le2_bn.most_probable_explanation({'J': True, 'O': False}))

#print(abc, inst)




# run the factor multiplication test

def test_factor_multiplication():
    """


    Parameters
    ----------
    factor1 : Dataframe CPT of first factor.
    factor2 : Dataframe CPT of second factor.

    Returns
    -------
    None. Prints out checks for a test case of factor multiplication

    """
    dog_example_path = 'testing/dog_problem.BIFXML'

    dog_bn = BNReasoner(dog_example_path)

    x = dog_bn.bn.get_all_cpts()

    print('TEST CASE: factor multiplication:')
    print('Testing with inputs Light on and Dog out:')
    print('Light on \n ----- \n')
    print(x['light-on'])
    print('Dog out \n ----- \n')
    print(x['dog-out'])
    print("\n Factor multiplication \n --------------- \n")
    asdf_factoreddf = dog_bn.factor_multiplication(x['light-on'], x['dog-out'])

    # Test output with predefined correct test case
    correctoutputs = [0.594, 0.006, 0.582,
                      0.018,
                      0.396,
                      0.004,
                      0.388,
                      0.012,
                      0.045000000000000005,
                      0.005000000000000001,
                      0.015,
                      0.034999999999999996,
                      0.855,
                      0.095,
                      0.285,
                      0.6649999999999999]
    incorrect_cases = []
    for i in range(len(asdf_factoreddf['p'])):
        if (correctoutputs[i] != asdf_factoreddf['p'][i]):
            incorrect_cases.append(asdf_factoreddf['p'][i])

    if not incorrect_cases:
        print('Factor Multiplication tested correct')
    else:
        print('Factor Multiplication tested incorrect')


test_factor_multiplication()

