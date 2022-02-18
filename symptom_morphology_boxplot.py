#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:56:46 2022

@author: ya-chenchuang
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import xlwt
from xlwt import Workbook

SymptomData = pd.read_excel('/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_CBLv.xls', sheet_name='CerebellumVolume')

Tonsilboxplot = SymptomData.boxplot(fontsize=None, rot=90)
Tonsilboxplot.set_ylabel("Cerebellum volume (mm^3)", fontsize = 14) 