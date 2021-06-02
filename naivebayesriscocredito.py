# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 22:38:28 2020

@author: julio
"""

import pandas as pd
base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transform = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[0, 1 , 2, 3])],remainder = 'passthrough')
previsores = column_transform.fit_transform(previsores)


#labelencoder_classe = LabelEncoder()
#classe = labelencoder_classe.fit_transform(classe)


from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

classificador.fit(previsores,classe)

#historia boa, divida alta, garantia nenhuma, renda>35
resultado = classificador.predict([[1,0,0,1,0,0,1,0,0,1]])

resultadoprob = classificador.predict_proba([[1,0,0,1,0,0,1,0,0,1]])

#historia ruim, divida alta, garantia adequada, renda<15
resultado = classificador.predict([[0,0,1,1,0,1,0,1,0,0]])

resultadoprob = classificador.predict_proba([[0,0,1,1,0,1,0,1,0,0]])

print(classificador.classes_)
print(classificador.class_prior_)