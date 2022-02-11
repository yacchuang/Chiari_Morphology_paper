# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:16:55 2022

@author: Ya-Chen Chuang
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from scipy.stats import mannwhitneyu, normaltest



df = pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/pythongraph/symptoms_correlation.xlsx", sheet_name='headache_TonsilV');



# statistics
yes = df.loc[(df.headache == "Y"), "TonsilVolume"].values
no = df.loc[(df.headache == "N"), "TonsilVolume"].values

log_yes = np.log(yes)
log_no = np.log(no)

stat_results = [mannwhitneyu(yes, no, alternative="two-sided")]
pvalues = [result.pvalue for result in stat_results]
print("Yes vs No: \n", stat_results, "\n")


# boxplot
boxplot = sns.boxplot(x="headache", y="TonsilVolume", data=df);
# boxplot = sns.boxplot(x="condition", y="4thVentricle", data=pd.melt(df), order=["Y", "N"])
# boxplot = sns.stripplot(x="condition", y="4thVentricle", data=pd.melt(df), marker="o", alpha=0.3, color="black", order=["Y", "N"])
boxplot.axes.set_title("headache vs TonsilVolume", fontsize=16)
boxplot.set_xlabel("headache", fontsize=14)
boxplot.set_ylabel("TonsilVolume", fontsize=14)


# statistical annotation
x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df['TonsilVolume'].max() + .1, .1, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)

plt.show()