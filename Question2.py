################################################################################
# Author: BO-YANG WU and PO-YING HUANG
# Team: bo-po
# Date: 04/25/2020
# Final Project
# Path 1 Question 2
# The city administration is cracking down on helmet laws, and wants to deploy
# police officers on days with high traffic to hand out citations. Can they use
# the next day's weather forecast to predict the number of bicyclists that day?
################################################################################

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")

###########################################################################################################
df= df.replace(to_replace = 'T', value = 0.05)
df= df.replace(to_replace = '0.47 (S)', value = 0.47)
df['Precipitation'] = df['Precipitation'].astype(float)* 30
df['Total'] = df['Total'].str.replace(',', '').astype(int)* 0.001

plot= df.plot()
plt.legend(fontsize=7, loc='upper left')
plt.title('Weather vs Total Bicycle')
plt.xlabel('Date', fontsize= 10)
plt.ylabel('y', fontsize= 10)
xlabels = ['', df['Date'][0], df['Date'][50], df['Date'][100], df['Date'][150], df['Date'][200]]
plot.set_xticklabels(xlabels)
plot.get_figure().savefig("Q2ALL.png", dpi= 200)
plt.show()


###########################################################################################################

df['Brooklyn Bridge'] = df['Brooklyn Bridge'].str.replace(',', '').astype(int)
df['Manhattan Bridge'] = df['Manhattan Bridge'].str.replace(',', '').astype(int)
df['Williamsburg Bridge'] = df['Williamsburg Bridge'].str.replace(',', '').astype(int)
df['Queensboro Bridge'] = df['Queensboro Bridge'].str.replace(',', '').astype(int)
df['Precipitation'] = df['Precipitation']/ 30
df['Total'] = df['Total']/ 0.001

degrees = [i + 1 for i in range (8)]

def feature_matrix(x , n):
    X = []
    for i in x:
        temp = [i ** j for j in range (n , -1 , -1)]
        X.append(temp)
    return X

def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    B= np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), y))
    return B

def mse(y , ybar):
    temp = 0
    for i in range(len(y)):
        temp += (np.absolute(y[i] - ybar[i])/ ybar[i]) ** 2
    return temp / len(y)
###########################################################################################################


y = list(df['Total'])
x = list(df['High Temp (°F)'])
#x = [list(df['High Temp (°F)']) , list(df['Low Temp (°F)']) , list(df['Precipitation'])]

x1 = []
y1 = []
for i in range(2 * len(x) // 3):
    x1.append(x[i])
    y1.append(y[i])
    
x = list(df['Low Temp (°F)'])
for i in range(2 * len(x) // 3):
    x1.append(x[i])
    y1.append(y[i])


paramFits = []
for j in degrees:
    
    X = feature_matrix(x1, j)
    B = least_squares(X , y1)
    paramFits.append(B)


plt.scatter(x1 , y1 , color = 'black' , label = 'data')
x1.sort()
for k in paramFits:
    n = len(k) - 1
    X = feature_matrix(x1, n)
    X = np.array(X)
    y_pred = np.matmul(X,k)
    plt.plot(x1, y_pred, label = 'n = '+ str(n))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Train')
plt.legend(loc='upper left')
plt.plot()
plt.savefig('Q2train.png', dpi=200, bbox_inches='tight')
plt.show()
###########################################################################################################

x = list(df['High Temp (°F)'])
xbar = []
y1 = []
for i in range(2 * len(x) // 3, len(x)):
    xbar.append(x[i])
    y1.append(y[i])

x = list(df['Low Temp (°F)'])
for i in range(2 * len(x) // 3, len(x)):
    xbar.append(x[i])
    y1.append(y[i])

for k in range(len(paramFits)):
    ybar = []
    for i in xbar:
        temp = 0
        for j in range(len(paramFits[k])):
            temp += i** (len(paramFits[k]) - j - 1) * paramFits[k][j]
        ybar.append(temp)
    
    plt.scatter(xbar , y1 , color = 'black' , label = 'a')
    plt.scatter(xbar , ybar , color = 'r' , label = 'ba')
    plt.xlabel('X', fontsize= 10)
    plt.ylabel('Y', fontsize= 10)
    plt.title('degree = ' + str(k+1))
    plt.plot()
    plt.savefig('Q2degree'+ str(k+1)+ '.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    ac = 0
    for i in range(len(y1)):
        ac = ac+ np.abs(y1[i] - ybar[i]) / y1[i]
    ac /= len(y1)
    ac = 1-ac
    
    m= 0
    for i in range(len(y1)):
        m= m+ ((np.absolute(y1[i]- ybar[i])/ y1[i])** 2)
    m= m/ len(y1)
    
    print('ac:', k+1, ac)
    print('mse', k+1, mse(y1, ybar), m)
    
###########################################################################################################
    
