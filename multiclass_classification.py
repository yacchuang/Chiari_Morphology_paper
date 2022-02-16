#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:28:02 2022

@author: ya-chenchuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

### Load data
df =  pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_correlation.xlsx", sheet_name='symptoms_volumetric');
print(df.shape) 
df.head(2)

### Symptoms data reshape to numbers
df.loc[df["headache"]=="N", "headache"]=0
df.loc[df["headache"]=="Y", "headache"]=1
df.loc[df["back_neck_ear_pain"]=="N", "back_neck_ear_pain"]=0
df.loc[df["back_neck_ear_pain"]=="Y", "back_neck_ear_pain"]=1
df.loc[df["nausea_vomit"]=="N", "nausea_vomit"]=0
df.loc[df["nausea_vomit"]=="Y", "nausea_vomit"]=1
print(df)

# Set X (1 morphometric result) and Y (multiple symptoms)
X = df['4thVentricle']
X = df["4thVentricle"].values.reshape(-1,1)
Y1 = df['headache']
Y2 = df['back_neck_ear_pain']
Y3 = df['nausea_vomit']
Y = [Y1, Y2, Y3]

# Set X scaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Multi-class Classification 
lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X, Y)

# Scatter plot
plt.scatter(X, Y1, color = "r")
plt.scatter(X, Y2, color = "b")
plt.scatter(X, Y3, color = "g")
plt.plot(X,lm.predict_proba(X)[:,1], color='red')

