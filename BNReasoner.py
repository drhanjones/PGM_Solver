from typing import Union
from BayesNet import BayesNet
from copy import deepcopy
import pandas as pd
import numpy as np

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
        """
        Parameters
        ----------
        f : first factor to multiply, type should be dataframe.
        g : second factor to multiply, type should be dataframe.

        Returns
        -------
        asdf_factoreddf :  dataframe of f and g multiplied, based on multiplying by matching True/False in columns.
        based on factor multiplication described in pg 4: https://www.cs.helsinki.fi/u/bmmalone/probabilistic-models-spring-2014/FactorElimination.pdf 
        
        Example
        -------
        factor_multiplication(cpts['light-on'],cpts['dog-out'])
        or
        factor_multiplication(cpts['light-on'],cpts['family-out'])
        """
        firstdf = f
        seconddf = g

        firstdf_headers = firstdf.columns.values.tolist()
        seconddf_headers = seconddf.columns.values.tolist()

        print('columns in f: ',firstdf_headers)
        print('columns in g: ',seconddf_headers)
        
        matching_cols = []
        for header in firstdf_headers:
            if header in  seconddf_headers:
                if header != 'p':
                    # print('Match found')
                    # print(header)
                    matching_cols.append(header)
        print('matching columns: ',matching_cols)
        
        firstmatchingcol_name = matching_cols[0]
        
        # get columns that factored df will contain
        array = np.array(firstdf_headers+seconddf_headers)
        unique = np.unique(array)
        # print('columns for factored df: ',unique)
    
        # initialize array to which we'll add factor multiplication values
        factoreddf = []
        factoreddf.append(unique)
        
        
        # print("printing each row")
        numrowsfirstdf, numcolsfirstdf = firstdf.shape
        numrowsseconddf, numcolsseconddf = seconddf.shape

        for rowinfirstdf in range(numrowsfirstdf):
            for rowinsecondtdf in range(numrowsseconddf):
                if (firstdf[firstmatchingcol_name][rowinfirstdf] == seconddf[firstmatchingcol_name][rowinsecondtdf]):
                    # print(firstdf['p'][rowinfirstdf]*seconddf['p'][rowinsecondtdf])
                    newrowtoappend = []
                    for thing in factoreddf[0]:
                        if thing != 'p':
                            if thing in firstdf_headers:
                                # print('factoreddf[',thing,']=',firstdf[thing][rowinfirstdf])
                                # print('factoreddf col',thing,firstdf[thing][rowinfirstdf])
                                newrowtoappend.append(firstdf[thing][rowinfirstdf])
                            elif thing in seconddf_headers:
                                # print('factoreddf col',thing,seconddf[thing][rowinsecondtdf])
                                newrowtoappend.append(seconddf[thing][rowinsecondtdf])
                    newrowtoappend.append(firstdf['p'][rowinfirstdf]*seconddf['p'][rowinsecondtdf])
                    # print('newrow: ',newrowtoappend)
                    factoreddf.append(newrowtoappend)
                            
        
        # print('factoreddf', factoreddf)
        asdf_factoreddf = pd.DataFrame(data = factoreddf[1:], 
                        columns = factoreddf[0])
        print('factoreddf as df: \n', asdf_factoreddf)
        return asdf_factoreddf



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
        #sum_column = "dog-out"
        gby_list = [e for e in list(df.columns) if e not in ['p', X]]
        df_ret = df.drop(X, axis=1).groupby(by=gby_list).sum().reset_index()

        return df_ret

    def maxing_out(self, X, f):
        df = f.copy(deep=True)
        gby_list = [e for e in list(df.columns) if e not in ['p', X]]

        idx = df_test.groupby(by=gby_list)['p'].transform(max) == df_test['p']
        df = df[idx].reset_index(drop=True)
        df_ret = df.drop(X, axis=1)

        return df_ret, df


dog_example_path = 'testing/dog_problem.BIFXML'

dog_bn = BNReasoner(dog_example_path)

x = dog_bn.bn.get_all_cpts()
# print(x)
#print(dog_bn.bn.draw_structure())

df_test = x["dog-out"].copy(deep=True)
sum_column = "dog-out"
gby_list = [e for e in list(df_test.columns) if e not in ['p',sum_column]]

df_test.drop(sum_column,axis=1).groupby(by=gby_list).sum().reset_index()

print("\n Factor multiplication \n --------------- \n")
dog_bn.factor_multiplication(x['light-on'],x['dog-out'])
