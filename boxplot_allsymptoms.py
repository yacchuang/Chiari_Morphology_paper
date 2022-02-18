#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:05:51 2022

@author: ya-chenchuang
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import xlwt
from xlwt import Workbook
# import xlrd, xlwt


### load data
df =  pd.read_excel("/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_correlation.xlsx", sheet_name='allsymptoms_volumetric', header = None, index_col = None);
print(df.shape) 
print("Column headings:")
print(df.columns)

### Create new worksheet
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('4thVentricleVolume')

col = 0
for symptoms in range(2,17):
    sheet1.write(0, col, df.loc[0, symptoms])
    col = col + 1


### classify symptoms data
# read morphometric data:
# set Y = 1; N = 0
for column in df[2:16]:
    df.loc[df[column]=="N", column]=0
    df.loc[df[column]=="Y", column]=1
    

# if symptom == Y read morphometric data
# if symptom == N skip the data

row = 0
col = 0
for column in range(2,17):
    for i in df.loc[:, column]:
        if i == 1:
            sheet1.write(row, col, df.iloc[row, 22])
            # print(df.iloc[row, 19])
        row = row + 1
    row = 0
    col = col + 1

# write the data in a new excel sheet 
wb.save('/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_4thVentricle.xls')

### make boxplot 
SymptomData = pd.read_excel('/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_4thVentricle.xls', sheet_name='4thVentricleVolume')

Tonsilboxplot = SymptomData.boxplot()  

