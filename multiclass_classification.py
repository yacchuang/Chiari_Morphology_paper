#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:28:02 2022

@author: ya-chenchuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

## Load data
df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/symptoms_morpho.xlsx", sheet_name='hydro_ICP');
print(df.shape) 
df.head(2)

## extra each variable
# tonsil = pd.DataFrame(df["TonsilV"].values, index = None, columns = ["TonsilV"]); 
# CBLv = pd.DataFrame(df["CBLv"].values, index = None, columns = ["CBLv"]); 
# BSv = pd.DataFrame(df["BSv"].values, index = None, columns = ["BSv"]); 
# Ventricle = pd.DataFrame(df["4thV"].values, index = None, columns = ["4thV"]); 
# tonsilL = pd.DataFrame(df["TonsilL"].values, index = None, columns = ["TonsilL"]);
# FMa = pd.DataFrame(df["FMaRatio"].values, index = None, columns = ["FMaRatio"]);
# Clivo_occipital = pd.DataFrame(df["Clivo_occipital"].values, index = None, columns = ["Clivo_ccipital"]);
# Boogard_Angle = pd.DataFrame(df["Boogard"].values, index = None, columns = ["Boogard"]);
# Occipital_angle = pd.DataFrame(df["Occipital"].values, index = None, columns = ["Occipital"]);
# Clivus_canal = pd.DataFrame(df["Clivus_canal"].values, index = None, columns = ["Clivus_canal"]);

df = df.dropna()
X = df.iloc[:,3:6].values


## label symptoms
# Chiari = df.loc[df["condition"]=="Chiari", "condition"]=0
# Healthy = df.loc[df["condition"]=="Healthy", "condition"]=1
label = df["symptoms"]
from sklearn.preprocessing import LabelEncoder 
ly = LabelEncoder()
y = ly.fit_transform(label)


# ## pairplot
# sns.set()
# sns.pairplot(df[['TonsilV', 'CBLv', 'BSv', '4thV', 'TonsilL', 'FMaRatio']], hue="condition", diag_kind="kde")

## Splitting Data using Sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


## Logistic Regression using Sklearn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs',multi_class='ovr', max_iter=100, C=1)
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)


acc1 = accuracy_score(y_test,y_pred)
p_pred1 = logreg.predict_proba(x_test)
y_pred1 = logreg.predict(x_test)
conf_m1 = confusion_matrix(y_test, y_pred)
report1 = classification_report(y_test, y_pred)


## Support Vector Machine using Sklearn
# from sklearn.svm import SVC
# svc1 = SVC(C=1,kernel='rbf',gamma=1)     
# svc1.fit(x_train,y_train)
# y_pred4 = svc1.predict(x_test)

# acc4= accuracy_score(y_test,y_pred4)
# # p_pred4 = svc1.predict_proba(x_test)
# y_pred4 = svc1.predict(x_test)
# conf_m4 = confusion_matrix(y_test, y_pred4)
# report4 = classification_report(y_test, y_pred4)

# # Scatter plot
# plt.scatter(x, y1, color = "r")
# plt.scatter(x, y2, color = "b")
# plt.scatter(x, y3, color = "g")
# plt.plot(X,lm.predict_proba(X)[:,1], color='red')

