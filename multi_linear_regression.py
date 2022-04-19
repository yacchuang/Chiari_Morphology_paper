#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:58:17 2022

@author: ya-chenchuang
"""

## Load in Libraries

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerLine2D

os.chdir('/Users/ya-chenchuang/Desktop/Stevens/projects/OL DRG coculture project/AFM')

tau2_3=pd.read_excel("tau1_data.xlsx",sheet_name="day3")

tau2_7=pd.read_excel("tau1_data.xlsx",sheet_name="day7")

tau2_14=pd.read_excel("tau1_data.xlsx",sheet_name="day14")


## Read Data

# Day 3
myelin_day3=tau2_3[["myelin"]]
myelin_day3["Treat"]="Myelin"
myelin_day3["Day"]=3
myelin_day3.columns=['E', 'Treat', 'Day']

unmyelin_day3=tau2_3[["unmyelin"]]
unmyelin_day3["Treat"]="Unmyelin"
unmyelin_day3["Day"]=3
unmyelin_day3.columns=['E', 'Treat', 'Day']

# Day 7
myelin_day7=tau2_7[["myelin"]]
myelin_day7["Treat"]="Myelin"
myelin_day7["Day"]=7
myelin_day7.columns=['E', 'Treat', 'Day']

unmyelin_day7=tau2_7[["unmyelin"]]
unmyelin_day7["Treat"]="Unmyelin"
unmyelin_day7["Day"]=7
unmyelin_day7.columns=['E', 'Treat', 'Day']

# Day 14
myelin_day14=tau2_14[["myelin"]]
myelin_day14["Treat"]="Myelin"
myelin_day14["Day"]=14
myelin_day14.columns=['E', 'Treat', 'Day']

unmyelin_day14=tau2_14[["unmyelin"]]
unmyelin_day14["Treat"]="Unmyelin"
unmyelin_day14["Day"]=14
unmyelin_day14.columns=['E', 'Treat', 'Day']

## Output Data
out_df=myelin_day3.append(unmyelin_day3)
out_df=out_df.append(myelin_day7)
out_df=out_df.append(myelin_day14)
out_df=out_df.append(unmyelin_day7)
out_df=out_df.append(unmyelin_day14)

out_df.dropna(inplace = True)

## Multiple linear regression
import statsmodels.formula.api as smf

# Initialise and fit linear regression model using `statsmodels`
model = smf.ols('E ~ Treat+Day', data=out_df)
model = model.fit()

print(model.params)
print(model.summary())

