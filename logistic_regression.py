#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 17:01:45 2022

@author: Ya-Chen Chuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns

### Load data
df =  pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/pythongraph/symptoms_correlation.xlsx", sheet_name='headache_TonsilL');
print(df.shape) 
df.head(2)


### Data reshape to numbers
X = df['TonsilLength']
y = df['headache']
# plt.scatter(X,y)
df.loc[df["headache"]=="N", "headache"]=0
df.loc[df["headache"]=="Y", "headache"]=1
X = df["TonsilLength"].values.reshape(-1,1)
y = df["headache"].values.reshape(-1,1)


### Model Selection
# Logistc Regression
LogR = LogisticRegression()
LogR.fit(X,np.ravel(y.astype(int)))


### matplotlib scatter funcion w/ logistic regression
plt.scatter(X,y)
plt.plot(X,LogR.predict_proba(X)[:,1], color='red')
plt.xlabel("TonsilLength (mm)")
plt.ylabel("Probability of Headache")


### Evaluate the model
p_pred = LogR.predict_proba(X)
y_pred = LogR.predict(X)
score = LogR.score(X, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

 