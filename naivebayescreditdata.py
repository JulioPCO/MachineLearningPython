# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 23:27:10 2020

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


from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
classificador.fit(previsores_treinamento,classe_treinamento)

resultado = classificador.predict(previsores_teste)

#comparação predição com resultado do teste
from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, resultado)

matriz = confusion_matrix(classe_teste,resultado)