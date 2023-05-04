# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:31:37 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel('C:/Users/ya-chen.chuang/Documents/symptoms_correlation.xlsx', sheet_name='symptoms_new2')

print(df.head())

## Extraxt relavent data to train
#surgery_morpho = ['Surgery', 'TonsilV', 'CBLv ', 'BSv', '4thVentricle', 'Tonsil length', '(CMa+Ta)/FMa', 'Clivo-occipital', 'Boogard Angle', 'Occipital angle', 'Clivus canal angle']
surgery_morpho = ['Surgery', 'Tonsil length', '(CMa+Ta)/FMa']
df_surgery = df[surgery_morpho]

df_surgery.Surgery[df_surgery.Surgery == "Y"] =1
df_surgery.Surgery[df_surgery.Surgery == "N"] =0

df_clean = df_surgery.dropna()

Y = df_clean['Surgery'].values
Y = Y.astype('int')

X = df_clean.drop(labels=['Surgery'], axis = 1)

## Train the data with Random Forest Algorithm
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10, random_state = 50)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
print("Accuracy=", metrics.accuracy_score(y_test, y_pred))


## Feature importancy list
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)