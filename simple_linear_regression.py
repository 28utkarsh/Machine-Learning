# Simple Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset= pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
"""
split = float(input('Enter the value to split:'))
length = len(dataset)
train_len = split*length
test_len = length - train_len
training_set = 
for i in range (0, length):
    if len(training_set) != train_len = 
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
sum_y = 0
sum_x = 0
sum_xy = 0
sum_xx = 0
prod_xy = 0
prod_xx = 0
for i in range(0,len(X_train)):
    sum_x = sum_x + X_train[i,0]
    print(X_train[i,0])
    sum_y = sum_y + y_train[i]
    prod_xy = X_train[i,0] * y_train[i]
    prod_xx = X_train[i,0]**2
    sum_xy = sum_xy + prod_xy
    sum_xx = sum_xx + prod_xx
print("Sum_X = "+str(sum_x)+"\nSum_Y ="+str(sum_y)+
      "\nSum_XY = "+str(sum_xy)+"\nSum_XX = "+str(sum_xx))

b = (sum_y * sum_xx - sum_x * sum_xy)/(len(X_train) * sum_xx - sum_x**2)
a = (len(X_train) * sum_xy - sum_x * sum_y)/(len(X_train)*sum_xx - sum_x**2)

itr = 0.1
step = 0.01
equ = []
sc = []
for i in range(0,1000):
    sc.append(itr)
    equ.append(a*itr+b)
    itr = itr + step

    
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(sc, equ, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()