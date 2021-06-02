# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 19:56:31 2020

@author: julio
"""

import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)


from sklearn.tree import DecisionTreeClassifier, export
import graphviz

classificacao = DecisionTreeClassifier(criterion='entropy',random_state=0)
classificacao.fit(previsores_treinamento,classe_treinamento)
previsoes = classificacao.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)


#Desenho arvore
export.export_graphviz(classificacao,
                       out_file = 'arvorecensus.dot')

#convert file dot to png
graphviz.render('dot', 'png', 'arvorecensus.dot')


import shutil

original = r'C:\Curso Machine Learning\ML Exercises\arvorecensus.dot'
target = r'C:\Curso Machine Learning\ML Exercises\Arquivos dot e png\arvorecensus.dot'

shutil.move(original,target)

original = r'C:\Curso Machine Learning\ML Exercises\arvorecensus.dot.png'
target = r'C:\Curso Machine Learning\ML Exercises\Arquivos dot e png\arvorecensus.dot.png'

shutil.move(original,target)