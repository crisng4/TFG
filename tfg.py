#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:02:41 2018

@author: cris
"""

from pandas.tools.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.decomposition import PCA

def csv(csv_file):
    pd_films = pd.read_csv(csv_file,sep = ',',header = 0)
    return pd_films

def findElement(lista, elemento):
    for i in range(0,len(lista)):
        if(lista[i] == elemento):
            return True
    
def noDuplicate(lista):
    lista_no_duplicate = set(lista)
    lista_no_duplicate=list(lista_no_duplicate)
    return lista_no_duplicate
    
def processGenres(genres):
    gen = []
    gen_aux = []
    for g in genres:
        gen.append(g.split('|'))
    for list_gen in gen:
        for g in list_gen:
            gen_aux.append(g) 
    aux = set(gen_aux)
    gen_aux=list(aux)
    return gen_aux

def getDummiesG(pd_films_no_numeric,pd_films_sin_nan,list_genres,ins):
    k = len(pd_films_no_numeric.columns)-1
    for col in list_genres:
       pd_films_no_numeric.insert(k,col,ins)
       k = k+1
    contador = 0
    pd_films_no_numeric.insert(1,'genres',pd_films_sin_nan['genres'])
    for list_gen in pd_films_no_numeric['genres']:
        print(contador,'de',len(pd_films_no_numeric))
        genero=[]
        genero = list_gen.split('|')
        for gen in list_genres:
            
            pd_films_no_numeric.loc[contador,gen]=1
        contador = contador + 1
    pd_films_no_numeric= pd_films_no_numeric.drop('genres', 1)
    return pd_films_no_numeric
       
def buildingMatrixNoNumeric(pd_films_sin_nan):
    columns = pd_films_sin_nan.columns
    columns_no_numeric_scpecific=[1,9,21]
    data = []
    pd_films_no_numeric = pd.DataFrame(data)
    i=0
    for col in columns_no_numeric_scpecific:
        print(i,'de',len(columns_no_numeric_scpecific))
        var = columns[col]
        pd_films_no_numeric.insert(i,var,pd_films_sin_nan[var])
        i=i+1
    var = columns[1]
    director = noDuplicate(pd_films_no_numeric[var])
    var = columns[9]
    genres = noDuplicate(pd_films_no_numeric[var])
    var = columns[21]
    content = noDuplicate(pd_films_no_numeric[var])
    list_genres = processGenres(genres)
    ins = np.zeros((len(pd_films_no_numeric), 1))
    pd_films_no_numeric= pd_films_no_numeric.drop('genres', 1)
    pd_films_no_numeric_3 =pd.get_dummies(pd_films_no_numeric)
    matrix_g = getDummiesG(pd_films_no_numeric_3,pd_films_sin_nan,list_genres,ins)
    var = 'movie_title'
    matrix_g.insert(0,var,pd_films_sin_nan[var])
    matrix_g = matrix_g.drop('movie_title', 1)
    return matrix_g
    
 
    
def prepocesingData(pd_films_csv):
    pd_films = pd_films_csv.copy()
    pd_films_numeric = pd_films.select_dtypes(include=['float64','int64'])
    matrix_no_numeric = buildingMatrixNoNumeric(pd_films)
    result = pd.concat([matrix_no_numeric,pd_films_numeric],axis=1)
    result = result.dropna(how='any')
    return result    

def getXAndY(matrix):
    x = matrix
    x = matrix.drop('imdb_score', 1)
    y = matrix['imdb_score']
    return x,y
    
def metrics(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    my_scaler = StandardScaler()
    X_train_scaled = my_scaler.fit(X_train)
    X_train_norm= my_scaler.transform(X_train)
    X_train_norm_sin_pca = X_train_norm.copy()
    my_pca = PCA()
    X_train_norm_pca = my_pca.fit(X_train_norm)
    X_train_norm_con_pca = my_pca.transform(X_train)
    # Regresión lineal sin PCA
    regression(X_train_norm_sin_pca,y_train,X_test,y_test,my_scaler)
    # Regresión lineal con PCA
    regression(X_train_norm_con_pca,y_train,X_test,y_test,my_scaler)
    
def  neuronalNetwork(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.25,random_state=0)
    #code to modify
    D = matrix.shape[1]
    
    #preprocessing (remember sin PCA)
    
    mlr = MLPRegressor((int(D/4.),int(D/16.),int(D/64)),activation = 'relu',alpha = 0,verbose = 4,tol = 0,learning_rate = 'adaptive',solver = 'sgd',momentum = 0.4,max_iter = 600)
    
    mlr.fit()
    
    #joblib.dump(grid, 'modelo2.pkl')
    y_hat = mlr.predict(x_scaled)
    
    rmse_train = np.sqrt(np.mean((y_train-y_hat)**2))
    rmse_test = np.sqrt(np.mean((y_test-y_hat)**2))

def regression(X_train_norm_sin_pca,y_train,X_test,y_test,my_scaler):
    rl= LinearRegression()
    rl.fit(X_train_norm_sin_pca,y_train)
    y_pred_training = rl.predict(X_train_norm_sin_pca)
    rmse_training = np.sqrt(mean_squared_error(y_train,y_pred_training))
    X_test_norm_sin_pca = my_scaler.transform(X_test)
    y_pred = rl.predict(X_test_norm_sin_pca)
    rmse_test= np.sqrt(mean_squared_error(y_test,y_pred))

    
if __name__ == '__main__':
    
    read_prepro_data = False
    
    if read_prepro_data:
        matrix = csv('movie_metadata.csv')
        result = prepocesingData(matrix)
        X,y = getXAndY(result)
        X = X.as_matrix()
        y = y.as_matrix()
        print("saving matrix")
        np.save('numpy_cine_X',X)
        np.save('numpy_cine_y',y)
    else:
        x = np.load('numpy_cine_X.npy')
        y = np.load('numpy_cine_y.npy')
    
    
    #metrics(result)
    neuronalNetwork(result)