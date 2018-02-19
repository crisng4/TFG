#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 02:05:16 2017

@author: cris
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 08:04:24 2017

@author: cris
"""
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv(csv_file):
     pd_films = pd.read_csv(csv_file,sep = ',',header = 0)
     columns = pd_films.columns
     rows = pd_films.index
     pd_films_sin_nan = pd_films.dropna(how='any')
     pd_films_numeric = pd_films_sin_nan.select_dtypes(include=['float64','int64'])
     columns3 = pd_films_numeric.columns
     #red(pd_films_numeric);
     return pd_films_numeric
     
print "hola"   
pd_films_numeric = read_csv('movie_metadata.csv')     
xx = pd_films_numeric.as_matrix()
print "hola" 
y = xx[:,13]
X = xx[:,range(13)+[14,15]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25,random_state=42)

tuning_param = {'hidden_layer_sizes':[[3,2],[2,3],[4,4,4],[3,3,3,3]],'activation':['logistic','relu'],'learning_rate_init':np.logspace(-2,3,30),'alpha':np.logspace(-4,4,30)}
#tuning_param = {'hidden_layer_sizes':[[3,2],[1,2]]}

mlp = MLPRegressor()

grid = GridSearchCV(mlp,param_grid = tuning_param, cv = 5,n_jobs = -1,verbose = 3)
my_pca = PCA()
my_pca.fit(X_train)
X_train_PCA = my_pca.transform(X_train)
grid.fit(X_train_PCA,y_train)

joblib.dump(grid, 'modelo2.pkl')

#if __name__ == '__main__':
#    read_csv('movie_metadata.csv')