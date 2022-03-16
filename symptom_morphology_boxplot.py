#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:56:46 2022

@author: ya-chenchuang
"""

import pandas as pd
import numpy as np


SymptomData = pd.read_excel('/Users/kurtlab/Desktop/Chiari Morphometric/results/symptoms_morph/symptoms_BSv.xls', sheet_name='BSv')

Tonsilboxplot = SymptomData.boxplot(fontsize=None, rot=90)
Tonsilboxplot.set_ylabel("Brainstem volume (mm^3)", fontsize = 14) 