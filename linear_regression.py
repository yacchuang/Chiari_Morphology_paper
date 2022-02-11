# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:11:07 2022

@author: Ya-Chen Chuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

# Load data
df =  pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/pythongraph/2D_3D_correlation.xlsx", sheet_name='Cere_Tonsil_volume');
print(df.shape) 
df.head(2)

# linear regression statistics
model_lin = sm.OLS.from_formula("Cere_Tonsil_volume ~ CereVolume", data=df)
result_lin = model_lin.fit()
result_lin.summary()
print(result_lin.summary())


