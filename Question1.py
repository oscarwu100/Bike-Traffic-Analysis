################################################################################
# Author: BO-YANG WU and PO-YING HUANG
# Team: bo-po
# Date: 04/25/2020
# Final Project
# Path 1 Question 1
# You want to install sensors on the bridges to estimate overall traffic across
# all the bridges. But you only have enough budget to install sensors on three of
# the four bridges. Which bridges should you install the sensors on to get the best
# prediction of overall traffic?
################################################################################

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr


df = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")
df= df.replace(to_replace = 'T', value = 0.05)
df= df.replace(to_replace = '0.47 (S)', value = 0.47)
df['Precipitation'] = df['Precipitation'].astype(float)
df['Brooklyn Bridge'] = df['Brooklyn Bridge'].str.replace(',', '').astype(int)
df['Manhattan Bridge'] = df['Manhattan Bridge'].str.replace(',', '').astype(int)
df['Williamsburg Bridge'] = df['Williamsburg Bridge'].str.replace(',', '').astype(int)
df['Queensboro Bridge'] = df['Queensboro Bridge'].str.replace(',', '').astype(int)
df['Total'] = df['Total'].str.replace(',', '').astype(int)


print("Pearson's correlation:")
# calculate Pearson's correlation

#Brooklyn Bridge
corr6, _ = pearsonr(df['Brooklyn Bridge'], df['Total'])
r6, _  = np.corrcoef(df['Brooklyn Bridge'], df['Total'])
b6= df['Brooklyn Bridge'].corr(df['Total'])
print('Brooklyn Bridge: ', b6)

#Manhattan Bridge
corr7, _ = pearsonr(df['Manhattan Bridge'], df['Total'])
r7, _  = np.corrcoef(df['Manhattan Bridge'], df['Total'])
b7= df['Manhattan Bridge'].corr(df['Total'])
print('Manhattan Bridge: ', b7)

#Williamsburg Bridge
corr8, _ = pearsonr(df['Williamsburg Bridge'], df['Total'])
r8, _  = np.corrcoef(df['Williamsburg Bridge'], df['Total'])
b8= df['Williamsburg Bridge'].corr(df['Total'])
print('Williamsburg Bridge: ', b8)

#Queensboro Bridge
corr9, _ = pearsonr(df['Queensboro Bridge'], df['Total'])
r9, _  = np.corrcoef(df['Queensboro Bridge'], df['Total'])
b9= df['Queensboro Bridge'].corr(df['Total'])
print('Queensboro Bridge: ', b9)



print()
print("Spearmanr's correlation:")
# calculate Spearmanr's correlation

#Brooklyn Bridge
s6, _  = spearmanr(df['Brooklyn Bridge'], df['Total'])
print('Brooklyn Bridge: ', s6)

#Manhattan Bridge
s7, _  = spearmanr(df['Manhattan Bridge'], df['Total'])
print('Manhattan Bridge: ', s7)

#Williamsburg Bridge
s8, _  = spearmanr(df['Williamsburg Bridge'], df['Total'])
print('Williamsburg Bridge: ', b8)

#Queensboro Bridge
s9, _  = spearmanr(df['Queensboro Bridge'], df['Total'])
print('Queensboro Bridge: ', b9)