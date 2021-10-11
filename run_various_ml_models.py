# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:34:48 2021

@author: bhupendra.singh
"""

# compare performance on the provided dataset using various ML models ############
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np


df = pd.read_csv("train_features.csv")
df = df.iloc[:, 1:]
x_train = df.to_numpy()

df = pd.read_csv("train_target.csv")
df = df.iloc[:, 1:]
y_train = df.to_numpy()

df = pd.read_csv("test_features.csv")
df = df.iloc[:, 1:]
x_test = df.to_numpy()

df = pd.read_csv("test_target.csv")
df = df.iloc[:, 1:]
y_test = df.to_numpy()

x_all = np.concatenate((x_train, x_test))

y_all = np.concatenate((y_train, y_test))

rus = RandomUnderSampler(random_state=42, replacement=True)# fit predictor and target variable
x_rus, y_rus = rus.fit_resample(x_all, y_all)

models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
cm =[]

for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_rus, y_rus, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
