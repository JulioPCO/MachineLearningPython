# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:15:53 2020

@author: julio
"""

import Orange
base = Orange.data.Table('census.csv')
base.domain

base_dividida = Orange.evaluation.testing.sample(base,n=0.25)

base_treinamento= base_dividida[1]

base_teste= base_dividida[0]


cn2_learner = Orange.classification.CN2Learner()
classificador = cn2_learner(base_treinamento)

#for regras in classificador.rule_list:
#    print(regras)

resultado = Orange.evaluation.TestOnTestData(base_treinamento,base_teste,[classificador])
print(Orange.evaluation.CA(resultado))