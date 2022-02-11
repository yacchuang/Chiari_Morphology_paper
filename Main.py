import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# DataLoc = '/Volumes/Kurtlab/aMRI/3DaMRIPaperResults/Amplified/ForPaper/PODResults/StatisticalResults/CerebellumDisplacementData.xlsx'
# DataLoc = '/Volumes/Kurtlab/aMRI/3DaMRIPaperResults/Amplified/ForPaper/PODResults/StatisticalResults/TonsilDisplacementData.xlsx'
DataLoc = '/Volumes/Kurtlab/aMRI/3DaMRIPaperResults/Amplified/ForPaper/PODResults/StatisticalResults/MedullaDisplacementData.xlsx'

dfHealthy = pd.read_excel(DataLoc, sheet_name='Healthy')
dfChiari = pd.read_excel(DataLoc, sheet_name='Chiari')

dfHealthy.dropna(inplace=True)
dfChiari.dropna(inplace=True)

def ParametrToAnalyze(dfHealthy,dfChiari,VariableName):


    HealthyBuilding = pd.DataFrame(dfHealthy[VariableName].values, columns=[VariableName])
    HealthyBuilding['HealthyOrChiari'] = 'Healthy'

    ChiariBuilding = pd.DataFrame(dfChiari[VariableName].values, columns=[VariableName])
    ChiariBuilding['HealthyOrChiari'] = 'Chiari'

    Combined = HealthyBuilding.append(ChiariBuilding)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 8)
    sns.set_theme()
    sns.set_style('white')
    ax = sns.boxplot(data=Combined, x="HealthyOrChiari", y=VariableName, palette=["b", "r"], linewidth=3)
    ax.grid(False)
    ax.tick_params(labelsize=35)
    # sns.set(font="cmr17")



VariableName = 'SI 90th'
ParametrToAnalyze(dfHealthy, dfChiari, VariableName)
x = 2