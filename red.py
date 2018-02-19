#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:19:11 2017

@author: cris
"""

from pandas.tools.plotting import scatter_matrix
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

pd_films = pd.read_csv('movie_metadata.csv',sep = ',',header = 0)
columns = pd_films.columns
rows = pd_films.index

pd_films_sin_nan = pd_films.dropna(how='any')
print pd_films_sin_nan



pd_films_sin_nan.describe()

columns2 =[]



#for i in columns:
#    print(i)
#    print(type(i))
#    print(i.dtype)
##    if type(i) in (int, float, long, complex):
##        columns2.append(i)
    

#for i in columns:
#    pd_films_sin_nan[1]
#pd_films_sin_nan.columns

pd_films_numeric = pd_films_sin_nan.select_dtypes(include=['float64','int64'])
pd_summary = pd_films_numeric.describe()
columns3 = pd_films_numeric.columns


X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
mlp = MLPClassifier(algorithm='l-bfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=60, cmap=mglearn.cm2)