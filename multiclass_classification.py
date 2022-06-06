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


# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


## Load data
# df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/symptoms_morpho.xlsx", sheet_name='SevereSymptom');
df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/morphology_results/morphometric_stat_combine.xlsx", sheet_name='2D3D_combine');
print(df.shape) 
df.head(2)

df = df.dropna()
# X = df.iloc[:,3:6].values


## Normalized Input
# feature_names = ["TonsilL", "(CMa+Ta)/FMa", "4thVentricle", "TonsilV", "CBLv", "BSv"]
feature_names = ["CBLv"]
for feature_name in feature_names:
    df[feature_name] = df[feature_name] / df[feature_name].std()
    
X = df[feature_names].values


## label symptoms
# Chiari = df.loc[df["condition"]=="Chiari", "condition"]=0
# Healthy = df.loc[df["condition"]=="Healthy", "condition"]=1
label = df["condition"]
# label = df["syringomyelia"]
from sklearn.preprocessing import LabelEncoder 
ly = LabelEncoder()
y = ly.fit_transform(label)


## pairplot
# sns.set()
# sns.pairplot(df[['TonsilV', 'CBLv', 'BSv', '4thV', 'TonsilL', 'FMaRatio']], hue="condition", diag_kind="kde")


## Splitting Data using Sklearn
from sklearn.model_selection import train_test_split
trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.5, random_state=2)


## Logistic Regression using Sklearn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs',multi_class='auto', max_iter=100, C=1)
logreg.fit(trainX,trainy)


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]

## Predict probabilities
lr_probs = logreg.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
score1 = logreg.score(trainX,trainy)
y_pred1 = logreg.predict(testX)


## Predict Probabilities
# score1 = logreg.score(X, y)
# pred1 = logreg.predict_proba(X)
# y_pred1 = logreg.predict(X)
# conf_m1 = confusion_matrix(X, y)
# report1 = classification_report(X, y)

# acc1 = accuracy_score(y_test,y_pred)
# p_pred1 = logreg.predict_proba(x_test)
# y_pred1 = logreg.predict(x_test)
# conf_m1 = confusion_matrix(y_test, y_pred)
# report1 = classification_report(y_test, y_pred)

## Logistic Regression Prediction
w0 = logreg.intercept_[0]
w = w1 = logreg.coef_[0]
 
equation = "y = %f + (%f * x1)  " % (w0, w1)

# w = w1, w2, w3, w4 = logreg.coef_[0]
 
# equation = "y = %f + (%f * x1) + (%f * x2) + (%f * x3) + (%f * x4)" % (w0, w1, w2, w3, w4)
print(equation)


# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


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

