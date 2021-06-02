# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:32:29 2020

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


from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p=2)

classificador.fit(previsores_treinamento,classe_treinamento)
previsões = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix , accuracy_score

precisão = accuracy_score(classe_teste,previsões)
matriz = confusion_matrix(classe_teste,previsões)
