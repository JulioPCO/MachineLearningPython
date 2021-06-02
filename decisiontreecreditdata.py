# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:14:42 2020

@author: julio
"""

import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


from sklearn.tree import DecisionTreeClassifier, export
import graphviz

classificador = DecisionTreeClassifier(criterion = "entropy",random_state = 0)

classificador.fit(previsores_treinamento,classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)


#Desenho arvore
export.export_graphviz(classificador,
                       out_file = 'arvorecreditdata.dot')

#convert file dot to png
graphviz.render('dot', 'png', 'arvorecreditdata.dot')


import shutil

original = r'C:\Curso Machine Learning\ML Exercises\arvorecreditdata.dot'
target = r'C:\Curso Machine Learning\ML Exercises\Arquivos dot e png\arvorecreditdata.dot'

shutil.move(original,target)

original = r'C:\Curso Machine Learning\ML Exercises\arvorecreditdata.dot.png'
target = r'C:\Curso Machine Learning\ML Exercises\Arquivos dot e png\arvorecreditdata.dot.png'

shutil.move(original,target)