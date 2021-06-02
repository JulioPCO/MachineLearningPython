# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:29:17 2020

@author: julio
"""

import pandas as pd
base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#column_transform = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[0, 1 , 2, 3])],remainder = 'passthrough')
#previsores = column_transform.fit_transform(previsores)

labelencoder_previsores = LabelEncoder()
previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder_previsores.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])

#labelencoder_classe = LabelEncoder()
#classe = labelencoder_classe.fit_transform(classe)


from sklearn.tree import DecisionTreeClassifier, export

classificador = DecisionTreeClassifier(criterion = "entropy")

classificador.fit(previsores,classe)

print(classificador.feature_importances_)

export.export_graphviz(classificador,
                       out_file ='arvore.dot',
                       feature_names = ['historia','divida','garantia','renda'],
                       class_names = ['Alto','Moderado','Baixo'],
                       filled = True,
                       leaves_parallel = True)

#teste
import graphviz
export.export_graphviz(classificador,
                       out_file ='arvoreteste.dot',
                       feature_names = ['historia','divida','garantia','renda'],
                       class_names = ['Alto','Moderado','Baixo'],
                       filled = True,
                       leaves_parallel = True)

#print tree
graphviz.Source.from_file("arvoreteste.dot")

#convert file dot to png
graphviz.render('dot', 'png', 'arvoreteste.dot')

#historia boa, divida alta, garantia nenhuma, renda>35
resultado = classificador.predict([[0,0,1,2]])

resultadoprob = classificador.predict_proba([[0,0,1,2]])

#historia ruim, divida alta, garantia adequada, renda<15
resultado = classificador.predict([[3,0,0,0]])

resultadoprob = classificador.predict_proba([[3,0,0,0]])

print(classificador.classes_)
