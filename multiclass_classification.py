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

from sklearn.preprocessing import StandardScaler

## Load data
df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/morphology_results/morphometric_stat_combine.xlsx", sheet_name='2D3D_combine');
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
X = df.iloc[:,3:13].values


## categorize Healthy vs Chiari
Chiari = df.loc[df["condition"]=="Chiari", "condition"]=0
Healthy = df.loc[df["condition"]=="Healthy", "condition"]=1
label = df["condition"]
# from sklearn.preprocessing import LabelEncoder 
# ly = LabelEncoder()
# y = ly.fit_transform(label)


## pairplot
# sns.set()
# sns.pairplot(df[['TonsilV', 'CBLv', 'BSv', '4thV', 'TonsilL', 'FMaRatio']], hue="condition", diag_kind="kde")

## Splitting Data using Sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,label,test_size=0.2)


## Logistic Regression using Sklearn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs',multi_class='auto')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_test,y_pred)


# ### Symptoms data reshape to numbers
# df.loc[df["NUMBNESS"]=="N", "NUMBNESS"]=0
# df.loc[df["NUMBNESS"]=="Y", "NUMBNESS"]=1
# df.loc[df["weakness"]=="N", "weakness"]=0
# df.loc[df["weakness"]=="Y", "weakness"]=2
# df.loc[df["gait_imbalance"]=="N", "gait_imbalance"]=0
# df.loc[df["gait_imbalance"]=="Y", "gait_imbalance"]=3
# print(df)

# # Set X (1 morphometric result) and Y (multiple symptoms)
# x = df['4thVentricle']
# x = df["4thVentricle"].values.reshape(-1,1)
# y1 = df['NUMBNESS']
# y2 = df['weakness']
# # y3 = df['gait_imbalance']
# Y = [y1, y2]
# X = [x, x]

# # Set X scaler
# sc = StandardScaler()
# x = sc.fit_transform(x)

# ### Model Selection
# # Multi-class Classification 
# lm = linear_model.LogisticRegression(multi_class='multinomial')
# lm.fit(X, Y)


# # Scatter plot
# plt.scatter(x, y1, color = "r")
# plt.scatter(x, y2, color = "b")
# # plt.scatter(x, y3, color = "g")
# plt.plot(X,lm.predict_proba(X)[:,1], color='red')

