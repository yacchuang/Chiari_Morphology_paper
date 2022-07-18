#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:50:57 2022

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
df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/symptoms_morpho.xlsx", sheet_name='surgery');
# df =  pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/morphology_results/morphometric_stat_combine.xlsx", sheet_name='2D3D_combine');
print(df.shape) 
df.head(2)

# df = df.dropna()



## Normalized Input
# feature_names = ["TonsilL", "(CMa+Ta)/FMa", "4thV", "TonsilV", "CBLv", "BSv", "Boogard", "Occipital"]
# feature_names = ["(CMa+Ta)/FMa", "4thVentricle", "TonsilV", "CBLv", "BSv", "Occipital"]
feature_names = ["BSv", "Occipital"]
variables = ["Surgery", "BSv", "Occipital"]
df_feature = df[variables].dropna()
for feature_name in feature_names:
    df_feature[feature_name] = df_feature[feature_name] / df_feature[feature_name].std()
    # df[feature_name] = df[feature_name] / df[feature_name].std()
    
X = df_feature[feature_names]
#X = df[feature_names].values


## label symptoms
# label = df_feature["syringomyelia"]
label = df_feature["Surgery"]
# label = df["syringomyelia"]
from sklearn.preprocessing import LabelEncoder 
ly = LabelEncoder()
y = ly.fit_transform(label)



## Logistic Regression using Sklearn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs',multi_class='auto', max_iter=100, C=1)
logreg.fit(X,y)


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y))]

## Predict probabilities
lr_probs = logreg.predict_proba(X)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
score1 = logreg.score(X,y)
y_pred1 = logreg.predict(X)



## Logistic Regression Prediction
w0 = logreg.intercept_[0]
w = w1, w2 = logreg.coef_[0]
 
equation = "y = %f + (%f * x1) + (%f * x2)" % (w0, w1, w2)

# w = w1, w2, w3, w4 = logreg.coef_[0]
 
# equation = "y = %f + (%f * x1) + (%f * x2) + (%f * x3) + (%f * x4)" % (w0, w1, w2, w3, w4)
print(equation)


# calculate scores
ns_auc = roc_auc_score(y, ns_probs)
lr_auc = roc_auc_score(y, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
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




