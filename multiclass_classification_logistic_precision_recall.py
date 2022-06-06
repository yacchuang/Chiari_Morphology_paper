#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:16:31 2022

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



## precision-recall curve and f1
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc



## Load data
df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/ChiariSymptoms_analysis.xlsx", sheet_name='syringomyelia_CM1');
# df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/morphology_results/morphometric_stat_combine.xlsx", sheet_name='2D3D_combine');
print(df.shape) 
df.head(2)

df = df.dropna()
# X = df.iloc[:,3:6].values


## Normalized Input
# feature_names = ["TonsilL", "(CMa+Ta)/FMa", "4thVentricle", "TonsilV", "CBLv", "BSv"]
feature_names = ["TonsilL", "(CMa+Ta)/FMa", "4thV"]
for feature_name in feature_names:
    df[feature_name] = df[feature_name] / df[feature_name].std()
    
X = df[feature_names].values


## label symptoms
label = df["syringomyelia"]
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
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, y_pred1), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

## Logistic Regression Equation
w0 = logreg.intercept_[0]
w = w1, w2 = logreg.coef_[0]
 
equation = "y = %f + (%f * x1) + (%f * x2) " % (w0, w1, w2)

# w = w1, w2, w3, w4 = logreg.coef_[0]
 
# equation = "y = %f + (%f * x1) + (%f * x2) + (%f * x3) + (%f * x4)" % (w0, w1, w2, w3, w4)
print(equation)





