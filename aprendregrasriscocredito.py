# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:57:22 2020

@author: julio
"""

import Orange

base = Orange.data.Table('risco_credito.csv')
base.domain

cn2_learner = Orange.classification.rules.CN2Learner()

classificador = cn2_learner(base)

for regras in classificador.rule_list:
    print(regras)
    
#historia boa, divida alta, garantia nenhuma, renda>35
#historia ruim, divida alta, garantia adequada, renda<15
    
resultado = classificador([['boa','alta','nenhuma','acima_35'],['ruim','alta','adequada','0_15']])
    
for i in resultado:
    print(base.domain.class_var.values[i])
    