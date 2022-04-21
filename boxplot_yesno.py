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



df = pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/symptoms_correlation_YN.xlsx", sheet_name='symptoms_new3D', index_col = None);



## statistics
yes = df.loc[(df.headache == "Y"), "CBLv"].values
no = df.loc[(df.headache == "N"), "CBLv"].values

# yes = df.loc[(df.sensory_changes == "Y"), "CBLv"].values
# no = df.loc[(df.sensory_changes == "N"), "CBLv"].values

# yes = df.loc[(df.nausea_vomit == "Y"), "CBLv"].values
# no = df.loc[(df.nausea_vomit == "N"), "CBLv"].values

# yes = df.loc[(df.gait_imbalance == "Y"), "CBLv"].values
# no = df.loc[(df.gait_imbalance == "N"), "CBLv"].values

# yes = df.loc[(df.numbness == "Y"), "CBLv"].values
# no = df.loc[(df.numbness == "N"), "CBLv"].values

# yes = df.loc[(df.dizziness == "Y"), "CBLv"].values
# no = df.loc[(df.dizziness == "N"), "CBLv"].values

# yes = df.loc[(df.spasm_hyperreflecxic_jerking_movement == "Y"), "CBLv"].values
# no = df.loc[(df.spasm_hyperreflecxic_jerking_movement == "N"), "CBLv"].values

# yes = df.loc[(df.tinnitus_vertigo == "Y"), "CBLv"].values
# no = df.loc[(df.tinnitus_vertigo == "N"), "CBLv"].values

# yes = df.loc[(df.seizures == "Y"), "CBLv"].values
# no = df.loc[(df.seizures == "N"), "CBLv"].values

# yes = df.loc[(df.urinary_bowel == "Y"), "CBLv"].values
# no = df.loc[(df.urinary_bowel == "N"), "CBLv"].values


log_yes = np.log(yes)
log_no = np.log(no)

stat_results = [mannwhitneyu(yes, no, alternative="two-sided")]
pvalues = [result.pvalue for result in stat_results]
print("Yes vs No: \n", stat_results, "\n")


# boxplot
boxplot = sns.boxplot(x="headache", y="CBLv", data=df);
# boxplot = sns.boxplot(x="condition", y="4thVentricle", data=pd.melt(df), order=["Y", "N"])
# boxplot = sns.stripplot(x="condition", y="4thVentricle", data=pd.melt(df), marker="o", alpha=0.3, color="black", order=["Y", "N"])
boxplot.axes.set_title("sheadache vs CBLv", fontsize=16)
boxplot.set_xlabel("headache", fontsize=14)
boxplot.set_ylabel("CBLv", fontsize=14)


# statistical annotation
x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df['CBLv'].max() + .1, .1, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)

plt.show()