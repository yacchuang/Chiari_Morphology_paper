# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 07:42:58 2022

@author: Ya-Chen Chuang
"""

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression
from scipy import stats

# Load data
df =  pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/pythongraph/2D_3D_correlation.xlsx", sheet_name='Chiari_3D');
print(df.shape) 
df.head(2)

# linear regression plot
X = df['4thVentricle']
y = df['Clivo_occipital']
# plt.scatter(X,y)
slope, intercept, r_value, p_value, std_err = stats.linregress(df['4thVentricle'],df['Clivo_occipital'])

sns.regplot(x="4thVentricle", y="Clivo_occipital", data = df,
       label="y={0:.1f}x+{1:.1f}".format(slope, intercept)).legend(loc="best")


plt.show()


