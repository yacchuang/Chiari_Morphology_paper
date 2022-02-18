#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:56:46 2022

@author: ya-chenchuang
"""

import pandas as pd
import numpy as np


SymptomData = pd.read_excel('/Users/ya-chenchuang/Desktop/Stevens/projects/Morphology/results/symptoms_Clivo-occipital.xls', sheet_name='Clivo-occipital')

Tonsilboxplot = SymptomData.boxplot(fontsize=None, rot=90)
Tonsilboxplot.set_ylabel("Clivo-occipital angle", fontsize = 14) 