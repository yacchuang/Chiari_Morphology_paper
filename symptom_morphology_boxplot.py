#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:56:46 2022

@author: ya-chenchuang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set the font globally
plt.rcParams.update({'font.family':'CMR17'})


SymptomData = pd.read_excel('/Users/kurtlab/Desktop/Chiari_Morphometric/results/symptoms_morph/symptoms_TonsilV.xls', sheet_name='Sheet1')

Tonsilboxplot = SymptomData.boxplot(fontsize=None, rot=90)
Tonsilboxplot.set_ylabel("Tonsil volume (mm^3)", fontsize = 14) 

plt.show()

