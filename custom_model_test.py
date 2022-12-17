#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 19:26:29 2022

@author: abishekthamma
"""
from BNReasoner import BNReasoner


custom_model_path = 'custom_model_task3.BIFXML'
custom_model_bn = BNReasoner(custom_model_path)

model_cpts = custom_model_bn.bn.get_all_cpts()
#print(model_cpts)

"""
model_cpts.keys()
['Plates moving away?', 'Magma level high?', 'Earthquake?', 
 'Volcanic Eruption?', 'Lateral blast?', 'Debris avalanche?', 
 'Aerial drops?', 'Tsunami?', 'Evacuation?', 'Take cover?'])
"""

#print(custom_model_bn.bn.draw_structure())

prior_marginal_q1 = custom_model_bn.marginal_distribution(['Evacuation?'])
print("Prior Marginal Query 1")
print(prior_marginal_q1)
prior_marginal_q2 = custom_model_bn.marginal_distribution(['Take cover?'])
print("Prior Marginal Query 2")
print(prior_marginal_q2)
posterior_marginal_q1 = custom_model_bn.marginal_distribution(['Evacuation?'], {'Lateral Blast?': True, 'Tsunami?': False, "Earthquake?":True})
print("Posterior Marginal Query 1")
print(posterior_marginal_q1)
MAP_query = custom_model_bn.maximum_a_posteriori(['Tsunami?'], {'Earthquake?': True, 'Volacano?': True, 'Lateral blast?': False})
print("MAP Query 1")
print(MAP_query)
MPE_query = custom_model_bn.most_probable_explanation({'Evacuation?': True})
print("MPE Query 1")
print(MPE_query.to_string())


"""

a prior marginal query: P(E'), P(C)
posterior marginal query: P(E|L=T,E=T,T=F)
MAP: Q={T}, Z={E=T, V=T, L=F}
MPE = Z={E'=T}
"""