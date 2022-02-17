# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:55:52 2021

@author: ME System 3
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu, normaltest

dfHealthy = pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/morphometric_stat_combine.xlsx", sheet_name='HealthyVolume');
dfChiari = pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/morphometric_stat_combine.xlsx", sheet_name='ChiariVolume');
VariableName = "4thVentricle"

HealthyVolume = pd.DataFrame(dfHealthy[VariableName].values, index = None, columns = [VariableName]); 
HealthyVolume['HealthyorChiari'] = "Healthy";

ChiariVolume = pd.DataFrame(dfChiari[VariableName].values, index = None, columns = [VariableName]); 
ChiariVolume['HealthyorChiari'] = "Chiari"; 


Compare = HealthyVolume.append(ChiariVolume);

# statistics
Healthy = dfHealthy['4thVentricle']
Chiari = dfChiari['4thVentricle']

log_Healthy = np.log(Healthy)
log_Chiari = np.log(Chiari)

stat_results = [mannwhitneyu(Healthy, Chiari, alternative="two-sided")]
pvalues = [result.pvalue for result in stat_results]
print("Healthy vs Chiari: \n", stat_results, "\n")


# statistical annotation
x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = dfHealthy['4thVentricle'].max() + 150, .1, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "****", ha='center', va='bottom', color=col)


# df = sns.load_dataset('test')
# df.head()

sns.set()
plot = sns.boxplot(x='HealthyorChiari', y = VariableName, data = Compare);
plot.set_ylabel("4th Ventricle volume (mm^3)", fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=14)
plt.show()