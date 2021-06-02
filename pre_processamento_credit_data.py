# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 08:35:38 2020

@author: julio
"""

import pandas as pd
base = pd.read_csv('credit_data.csv')

base.describe() #give some statistics about the database

base.loc[base['age'] < 0] #locate

#erase columnn

base.drop('age',1,inplace=True)

#erase registers with problem

base.drop(base[base.age<0].index,inplace=True)

#fill values manually
#fill values with mean
base.mean()
base['age'].mean()
meanbaseage = base['age'][base.age>0].mean()
base.loc[base.age<0,'age'] = meanbaseage

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:,1:4].values #coluna 4 não entra
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
Imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
Imputer = Imputer.fit(previsores[:,0:3])
previsores[:,0:3] = Imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)