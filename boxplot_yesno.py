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
import xlsxwriter
from xlwt import Workbook




df = pd.read_excel("/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/symptoms_morpho.xlsx", sheet_name='surgery', header = None, index_col = None);


df.loc[df[0]=="N", 0]=0
df.loc[df[0]=="Y", 0]=1
### Create new worksheet
wb = xlsxwriter.Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_worksheet('TonsilV')

# BackpainYes = df.loc[(df.back_neck_ear_pain == "Y"), "CBLv"].values
# BackpainNo = df.loc[(df.back_neck_ear_pain == "N"), "CBLv"].values

TonsilVYes = df.loc[(df[0] == 1), 2].values
TonsilVNo = df.loc[(df[0] == 0), 2].values

'''
row = 0
col = 0
for TonsilVYes, TonsilVNo in TonsilVYes, TonsilVNo:
    sheet1.write(row, col, TonsilVYes)
    sheet1.write(row, col+1, TonsilVNo)
    row += 1
'''

# yes = df.loc[(df.nausea_vomit == "Y"), "CBLv"].values
# no = df.loc[(df.nausea_vomit == "N"), "CBLv"].values

CBLvYes = df.loc[(df[0] == 1), 3].values
CBLvNo = df.loc[(df[0] == 0), 3].values

BSvYes = df.loc[(df[0] == 1), 4].values
BSvNo = df.loc[(df[0] == 0), 4].values

# DizzinessYes = df.loc[(df.dizziness == "Y"), "CBLv"].values
# DizzinessNo = df.loc[(df.dizziness == "N"), "CBLv"].values

ventricleYes = df.loc[(df[0] == 1), 5].values
ventricleNo = df.loc[(df[0] == 0), 5].values

TLYes = df.loc[(df[0] == 1), 7].values
TLNo = df.loc[(df[0] == 0), 7].values

FMaYes = df.loc[(df[0] == 1), 8].values
FMaNo = df.loc[(df[0] == 0), 8].values

OccipitalYes = df.loc[(df[0] == 1), 12].values
OccipitlNo = df.loc[(df[0] == 0), 12].values

# SpasmYes = df.loc[(df.Surgery == "Y"), "CBLv"].values
# SpasmNo = df.loc[(df.Surgery == "N"), "CBLv"].values

# yes = df.loc[(df.tinnitus_vertigo == "Y"), "CBLv"].values
# no = df.loc[(df.tinnitus_vertigo == "N"), "CBLv"].values

# yes = df.loc[(df.seizures == "Y"), "CBLv"].values
# no = df.loc[(df.seizures == "N"), "CBLv"].values

# yes = df.loc[(df.urinary_bowel == "Y"), "CBLv"].values
# no = df.loc[(df.urinary_bowel == "N"), "CBLv"].values

# write the data in a new excel sheet 
sheet1.write(TonsilVYes, TonsilVNo)
wb.save('/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/surgery_volume.xls')

## statistics
log_yes = np.log(yes)
log_no = np.log(no)

stat_results = [mannwhitneyu(yes, no, alternative="two-sided")]
pvalues = [result.pvalue for result in stat_results]
print("Yes vs No: \n", stat_results, "\n")


# boxplot
boxplot = sns.boxplot(x="Surgery", y="CBLv", data=df);
# boxplot = sns.boxplot(x="condition", y="4thVentricle", data=pd.melt(df), order=["Y", "N"])
# boxplot = sns.stripplot(x="condition", y="4thVentricle", data=pd.melt(df), marker="o", alpha=0.3, color="black", order=["Y", "N"])
boxplot.axes.set_title("headache vs CBLv", fontsize=16)
boxplot.set_xlabel("headache", fontsize=14)
boxplot.set_ylabel("CBLv", fontsize=14)


# statistical annotation
x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df['CBLv'].max() + .1, .1, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)

plt.show()