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
# df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/ChiariSymptoms_analysis.xlsx", sheet_name='syringomyelia');
df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/morphology_results/morphometric_stat_combine.xlsx", sheet_name='2D3D_combine');
print(df.shape) 
df.head(2)

df = df.dropna()
# X = df.iloc[:,3:6].values


## Normalized Input
# feature_names = ["TonsilL", "ratio", "ventricle", "TonsilV", "CBLv", "BSv", "Boogard", "Occipital"]
feature_names = ["TonsilV", "CBLv"]
for feature_name in feature_names:
    df[feature_name] = df[feature_name] / df[feature_name].std()
    
X = df[feature_names].values


## label symptoms
label = df["condition"]
# label = df["syringomyelia"]
from sklearn.preprocessing import LabelEncoder 
ly = LabelEncoder()
y = ly.fit_transform(label)



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



## Logistic Regression Prediction
w0 = logreg.intercept_[0]
w = w1, w2 = logreg.coef_[0]
 
equation = "y = %f + (%f * x1) + (%f * x2) " % (w0, w1, w2)

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


