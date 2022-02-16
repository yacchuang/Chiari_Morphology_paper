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
# from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns

### Load data
df =  pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_correlation.xlsx", sheet_name='symptoms_volumetric');
print(df.shape) 
df.head(2)


### Data reshape to numbers
X = df['4thVentricle']
y = df['headache']
# plt.scatter(X,y)
df.loc[df["headache"]=="N", "headache"]=0
df.loc[df["headache"]=="Y", "headache"]=1
X = df["4thVentricle"].values.reshape(-1,1)
y = df["headache"].values.reshape(-1,1)

# Create an instance of the scaler and apply it to the data
sc = StandardScaler()
X = sc.fit_transform(X)

### Model Selection
# Logistc Regression
LogR = LogisticRegression()
LogR.fit(X,np.ravel(y.astype(int)))


### matplotlib scatter funcion w/ logistic regression
plt.scatter(X,y)
plt.plot(X,LogR.predict_proba(X)[:,1], color='red')
plt.xlabel("4thVentricle (mm^3)")
plt.ylabel("Probability of Headache")


### Evaluate the model
p_pred = LogR.predict_proba(X)
y_pred = LogR.predict(X)
score = LogR.score(X, y)
acc =  accuracy_score(y, y_pred)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

 