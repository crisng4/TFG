#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:02:41 2018

@author: cris
"""

from pandas.tools.plotting import scatter_matrix
import pandas as pd
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.decomposition import PCA

def csv(csv_file):
    pd_films = pd.read_csv(csv_file,sep = ',',header = 0)
    return pd_films

    
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
    
 
    
def preprocesingData(pd_films_csv):
    pd_films = pd_films_csv.copy()
    pd_films_numeric = pd_films.select_dtypes(include=['float64','int64'])
    pd_films_numeric_nan= pd_films_numeric.dropna(how='any')
    matrix_no_numeric = buildingMatrixNoNumeric(pd_films)
    result = pd.concat([matrix_no_numeric,pd_films_numeric],axis=1)
    result = result.dropna(how='any')
    return result, pd_films_numeric_nan   

def explo_analysis(pd_films_numeric):
    columns = pd_films_numeric.columns
    rows = pd_films_numeric.index
    score = columns[13]
    i = 1
    j = 1

    for column in columns:
        plt.figure(i)    
        pd_films_numeric[column].plot(kind="hist",color='b',linestyle = '-')
        plt.title("Histograma " + column)
        plt.xlabel(column)
        ax = plt.axes()
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.show()
        i+=1
    
    for column in columns:
        plt.figure(j)
        ax = pd_films_numeric.plot(kind = 'scatter',x= column,y = 'imdb_score')
        ax.set_xscale('log')
        plt.title('imdb_score'+ ' VS ' + column)
        plt.show()
        j+=1
     
    sm=scatter_matrix(pd_films_numeric, alpha = 0.2, figsize = (16,16), diagonal = 'kde')

    [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(45) for s in sm.reshape(-1)]

    [s.get_xaxis().set_label_coords(0,-1) for s in sm.reshape(-1)]
    [s.get_yaxis().set_label_coords(-1,0) for s in sm.reshape(-1)]
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]

    plt.show()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    f, ax = plt.subplots(figsize=(16, 16))
    corr = pd_films_numeric.corr()
    ax = sns.heatmap(corr,cmap="YlGnBu",annot=True)
    ax.set_xlabel('Matriz de correlaciones', rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
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
    X_test_norm= my_scaler.transform(X_test)
    X_train_norm_sin_pca = X_train_norm.copy()
    X_test_norm_sin_pca = X_test_norm.copy()
    my_pca = PCA()
    X_train_norm_con_pca = my_pca.fit_transform(X_train_norm)
    X_test_norm_con_pca = my_pca.transform(X_test_norm)
    # Regresión lineal sin PCA
    regression(X_train_norm_sin_pca,y_train,X_test_norm_sin_pca,y_test,my_scaler,method = 'No PCA')
    # Regresión lineal con PCA
    regression(X_train_norm_con_pca,y_train,X_test_norm_con_pca,y_test,my_scaler,method = 'PCA')

def regression(X_train_norm,y_train,X_test,y_test,my_scaler,method):
    rl= LinearRegression()  
    rl.fit(X_train_norm,y_train)
    y_pred_training = rl.predict(X_train_norm)
    #results
    rmse_train = np.sqrt(mean_squared_error(y_train,y_pred_training))
    print "RMSE train LR ("+ method +") : "+str(rmse_train)
    print "R2 train LR ("+ method +") : "+str(r2_score(y_train,y_pred_training))
    
    y_pred_test= rl.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
    print "RMSE test LR ("+ method +") : "+str(rmse_test)
    print "R2 test LR ("+ method +") : "+str(r2_score(y_test,y_pred_test))
    
    
    
def  neuralNetwork(x,y,load):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.25,random_state=0)
    D= x.shape[1]
    my_scaler = StandardScaler()
    X_train_scaled = my_scaler.fit(X_train)
    X_train_norm= my_scaler.transform(X_train)
    paramgrid = {'alpha' : np.logspace(-4,3,15)}
     
    if load:
        clf = joblib.load('mlr_alphas.pkl')
    else:
        mlr = MLPRegressor((int(D/4.),int(D/16.),int(D/64)),activation = 'relu',verbose=1,tol = 0,learning_rate = 'adaptive',solver = 'sgd',momentum = 0.4,max_iter = 1000)
        clf = GridSearchCV(mlr,paramgrid,verbose = 4)
        clf.fit(X_train_norm,y_train)
        joblib.dump(clf,'mlr_alphas.pkl')
    #results
    y_hat = clf.predict(X_train_norm)
    rmse_train = np.sqrt(np.mean((y_train-y_hat)**2))
    print "RMSE train NN:"+ str(rmse_train)
    print 'R2 train NN: '+str(r2_score(y_train,y_hat))
    
    X_test_n = my_scaler.transform(X_test)
    y_hat_test = clf.predict(X_test_n)
    rmse_test = np.sqrt(mean_squared_error(y_test,y_hat_test))
    print"RMSE test NN: "+ str(rmse_test)
    print'R2 test NN:'+ str(r2_score(y_test,y_hat_test))
    ml  = clf.best_estimator_
    plt.plot(ml.loss_curve_)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    return my_scaler,clf,X_train,X_test,y_train,y_test

if __name__ == '__main__':
    
    value_input = raw_input('Read csv? y/n: ')
    if value_input=='y':
        read_prepro_data= True
        load = False
    elif value_input=='n':
        read_prepro_data= False
        load = True
    else:
        print ("Invalid syntax")
        sys.exit()
        
    if read_prepro_data:
        matrix = csv('movie_metadata.csv')
        result, pd_films_numeric_nan = preprocesingData(matrix)
        #explo_analysis(pd_films_numeric_nan)
        X,y = getXAndY(result)
        X = X.as_matrix()
        y = y.as_matrix()
        print("Saving matrix")
        #np.save('matri')
        np.save('numpy_cine_X',X)
        np.save('numpy_cine_y',y)
    else:
        x = np.load('numpy_cine_X.npy')
        y = np.load('numpy_cine_y.npy')
    
    value_input2 = raw_input('Linear Regression = 1| Neural Network = 2: ')
    if value_input2== '1':
        if read_prepro_data:
            x = np.load('numpy_cine_X.npy')
            y = np.load('numpy_cine_y.npy')
        metrics(x,y) #Linear Regression
    elif value_input2=='2':
        if read_prepro_data:
            x = np.load('numpy_cine_X.npy')
            y = np.load('numpy_cine_y.npy')
        my_scaler,mlr,X_train,X_test,y_train,y_test = neuralNetwork(x,y,load) #Neuronal Network
    else:
        print ("Invalid syntax")
        sys.exit()
  
    