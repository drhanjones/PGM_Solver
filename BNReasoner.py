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
        gby_list = [e for e in list(df.columns) if e not in ['p', X]]

        idx = df.groupby(by=gby_list)['p'].transform(max) == df['p']
        df = df[idx].reset_index(drop=True)
        print(df)
        print(X)
        instantiation = {X: df[X][0]}

        df_ret = df.drop(X, axis=1)

        return df_ret, instantiation


    def ordering(self, X):

        ordering_set = set(self.bn.get_all_variables()) - set(X)
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
    def variable_elimination(self, X, cpt_updated_evidence, elim_type = 'sumout'):

        ordering_set = self.ordering(X)
        #print(ordering_set)
        current_cpt_list = deepcopy(cpt_updated_evidence)
        if elim_type == 'maxout':
            instantiation_dict = {}
        for var in ordering_set:
            sum_out_list = [x for x in current_cpt_list.keys() if var in current_cpt_list[x].columns]
            #print("12",sum_out_list)
            if elim_type == 'sumout':
                joint_df = self.factor_multiplication_multiple([current_cpt_list[x] for x in sum_out_list])
                df_wo_var = self.calulate_marginalisation(var, joint_df)
                #print("123", df_wo_var)
            for d_var in sum_out_list:
                current_cpt_list.pop(d_var)
                if df_wo_var is not None:
                    current_cpt_list['wo_'+var] = df_wo_var

            """
            sum_out_a = current_cpt_list[sum_out_list[0]]
            for sum_out_b in sum_out_list[1:]:
                #print("33",sum_out_b)
                sum_out_a = self.factor_multiplication(sum_out_a, current_cpt_list[sum_out_b])
                if elim_type == 'sumout':
                    df_wo_var = self.calulate_marginalisation(var, sum_out_a)
                elif elim_type == 'maxout':
                    df_wo_var, instantiation = self.maxing_out(var, sum_out_a)
                    instantiation_dict = {**instantiation_dict, **instantiation}
            """
        #print(current_cpt_list)
        final_joint_df = self.factor_multiplication_multiple(list(current_cpt_list.values()))
            
        #print(current_cpt_list)
        if elim_type == 'sumout':
            return final_joint_df
        elif elim_type == 'maxout':
            return df_wo_var, instantiation_dict


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

        joint_marginal_df = self.variable_elimination(Q, evidence_cpt_list, elim_type = 'sumout')

        return joint_marginal_df

    def marginal_distribution(self, Q, e=None):

        marginals_df = self.joint_marginal(Q, evidence=e)
        marginals_df['p'] = marginals_df['p']/marginals_df['p'].sum()

        return marginals_df

    def maximum_a_posteriori(self, Q, e):

        default_cpt_list = self.bn.get_all_cpts()
        evidence_cpt_list = self.update_cpt_with_evidence(default_cpt_list, e)
        #maxout_return, instantiation = self.variable_elimination(Q, evidence_cpt_list, elim_type='maxout')
        marginals_df = self.joint_marginal(Q, evidence=e)
        print(marginals_df)
        #return maxout_return, instantiation

    def most_probability_explanation(self, e):

        default_cpt_list = self.bn.get_all_cpts()
        evidence_cpt_list = self.update_cpt_with_evidence(default_cpt_list, e)





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
abc = le2_bn.marginal_distribution(["J","O"])
print(abc)
#abc, inst = le2_bn.maximum_a_posteriori(['I',"J"], {'O': True})
#print(abc, inst)
#%%


"""
    def factor_multiplication(self, f, g):
        """"""
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
        """"""
        firstdf = f
        seconddf = g
        
        print(firstdf, seconddf)
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
        print('columns for factored df: ',unique)
    
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
        #print('factoreddf as df: \n', asdf_factoreddf)
        return asdf_factoreddf





"""