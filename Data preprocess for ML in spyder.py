#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:00:43 2023

@author: myyntiimac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/myyntiimac/Desktop/ML Datar/Data.csv")
df.describe()
df.shape
df.columns
df.head()
len(df.columns)
df.isnull()
X = df.iloc[:, :-1].values	
y= df.iloc[:,3].values
#imputation,filling missing value with mean, median mode
#by simpleimputer function from sklearn.impute, need to define imputer function 
#then fit with target column and trnsform it
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])
X[:, 1:3]
#lebelencoding to catagorical value, that labeled the catgorical variable by dummy variable like 1,0
#by lablelencoder function , first define the function and then fit_transform with target variable
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0]) 

X[:,0]=labelencoder_X.fit_transform(X[:,0]) #parameter of x
X[:,0]
#lebelencoding target variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)#parameter of y
y
#split the train and test set by train_test split function which imported from 
#sklearn.modelselection, in parameter in function , need to mention test size and random state)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state = 0 ) 

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3) 
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state = 0) 
#Feature scaling, its need to normalize the data, because some variable is range is high some small
from sklearn.preprocessing import Normalizer

sc_X = Normalizer() 

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)


