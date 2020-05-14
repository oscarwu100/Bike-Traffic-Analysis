################################################################################
# Author: BO-YANG WU and PO-YING HUANG
# Team: bo-po
# Date: 04/25/2020
# Final Project
# Path 1 Question 3
# Can you use this data to predict whether it is raining based on the number of bicyclists on the bridges?
################################################################################

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


df = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")
df= df.replace(to_replace = 'T', value = 0.05)
df= df.replace(to_replace = '0.47 (S)', value = 0.47)
df['Precipitation'] = df['Precipitation'].astype(float)
df['Brooklyn Bridge'] = df['Brooklyn Bridge'].str.replace(',', '').astype(int)
df['Manhattan Bridge'] = df['Manhattan Bridge'].str.replace(',', '').astype(int)
df['Williamsburg Bridge'] = df['Williamsburg Bridge'].str.replace(',', '').astype(int)
df['Queensboro Bridge'] = df['Queensboro Bridge'].str.replace(',', '').astype(int)
df['Total'] = df['Total'].str.replace(',', '').astype(int)



# calculate Pearson's correlation
p= []
t= []
for i in range(len(df['Precipitation'])):
    if df['Total'][i]> 20000:
        p.append(df['Precipitation'][i])
        t.append(df['Total'][i])

n, _= np.corrcoef(t, p)
print('Keep: ', n)


print("Pearson's correlation:")
#Brooklyn Bridge
corr6, _ = pearsonr(df['Precipitation'], df['Total'])
r6, _  = np.corrcoef(df['Precipitation'], df['Total'])
b6= df['Precipitation'].corr(df['Total'])
print('Precipitation: ', b6)


print()
print("Spearmanr's correlation:")
# calculate Spearmanr's correlation

#Brooklyn Bridge
s6, _  = spearmanr(df['Precipitation'], df['Total'])
print('Precipitation: ', s6)

y = list(df['Total'])
x = list(df['Precipitation'])
plt.title('Total vs Precipitation', fontsize= 10)
plt.xlabel('Precipitation', fontsize= 10)
plt.ylabel('Total', fontsize= 10)
plt.scatter(x , y , color = 'black' , label = 'data')
plt.savefig('Q3.png', dpi=200, bbox_inches='tight')










