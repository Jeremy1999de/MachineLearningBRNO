#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 08:10:20 2023

@author: desaillyjeremy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])
# Calculate the covariance matrix:
R=np.matmul(X,X.T)/3
print(R)
# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]

Xi1=np.matmul(np.transpose(X),u1)
Xi2=np.matmul(np.transpose(X),u2)
# Calculate the coordinates in new orthonormal basis:
Xaprox=np.matmul(u1[:,None],Xi1[None,:])#+np.matmul(u2[:,None],Xi2[None,:])


# Calculate the approximation of the original from new basis
#print(Xi1[:,None]) # add second dimention to array and test it


# Check that you got the original


# Load Iris dataset as in the last PC lab:
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[:])
     
# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show


# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)

Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)
#print(np.mean(Xpp[:,0]))

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show


# Compute pca.explained_variance_ and pca.explained_cariance_ratio_values
pca.explained_variance_


pca.explained_variance_ratio_


# Plot the principal components in 2D, mark different targets in color
plt.figure()
plt.scatter(Xpca[y==0,0],Xpca[y==0,1],color='green')
plt.scatter(Xpca[y==1,0],Xpca[y==1,1],color='blue')
plt.scatter(Xpca[y==2,0],Xpca[y==2,1],color='red')
plt.show()

# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred=knn1.predict(X_test)
# Import and show confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
confusion_matrix(y_test,Ypred)
ConfusionMatrixDisplay.from_predictions(y_test,Ypred)
fig=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,Ypred))
fig.plot()
plt.show()


# Now do the same, but use only 2-dimensional data of original X (first two columns)
X_trainWrong, X_testWrong, y_trainWrong, y_testWrong=train_test_split(X[:,0:1],y,test_size=0.3)
knn1=KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_trainWrong, y_trainWrong)
YpredWrong=knn1.predict(X_testWrong)
ConfusionMatrixDisplay.from_predictions(y_testWrong, YpredWrong)


