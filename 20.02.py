#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:55:44 2023

@author: desaillyjeremy
"""

# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

#We load IRIS dataset 
from sklearn.datasets import load_iris
#We store it in the iris variable
iris=load_iris()    
#An array that contains the names of the four features in the Iris dataset.                             
iris.feature_names

#Prints out the names of the features
print("Names of the features", iris.feature_names)
#Prints out the first five rows of the data matrix
print("First five rows", iris.data[0:5,:])
#Prints out the first five values of the target variable
print("First five target variable", iris.target[0:5])
#print(iris.data)

#We split data into training and testing parts:
from sklearn.model_selection import train_test_split
#We assign the data matrix from the Iris dataset to the X variable.
X=iris.data[iris.target!=2,0:2]
#We assign the data matrix from the Iris dataset to the y variable.
y=iris.target[iris.target!=2]
#Training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#The training matrix will have 120 rows and 4 columns
#The test matrix will have 30 rows and 4 columns
print(" X train shape", X_train.shape)
print("X test shape", X_test.shape)

#We use a Support Vector Machine for classification:
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
print("SVM Model score", SVMmodel.score(X_test,y_test))
print("SVM Model parameters", SVMmodel.get_params())

#We choose only first two features (columns) of iris.data' and we eliminate iris.target =2 from the data


supvectors=SVMmodel.support_vectors_
print("Supvectors shape", supvectors.shape)
print(" X train shape", X_train.shape)
print("Y train shape", y_train.shape)

print("Supvectors", supvectors)
#We plot scatterplots of targets 0 and 1 and check the separability of the classes:
plt.scatter(X[y==0,0],X[y==0,1],color='green')
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
plt.scatter(X[y==2,0],X[y==2,1],color='cyan')
plt.scatter(supvectors[:,0],supvectors[:,1],color='red',marker='+',s=50)

W=SVMmodel.coef_
b=SVMmodel.intercept_
xgr=np.linspace(min(X[:,0]),max(X[:,0]),100)



print(W[:,0])
print(W[:,1])
print(b)
ygr=-W[:,0]/W[:,1]*xgr-b/W[:,1]
plt.scatter(xgr,ygr)



""" Exercice 2 """
print(" ----------- Exercice 2 ----------- ")

from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random

plt.figure()
random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))

plt.scatter(x[:,0], x[:,1])
plt.show()


SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)

SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]

plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()











