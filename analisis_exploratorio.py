#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:04:54 2017

@author: cris
"""
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import matplotlib.pyplot as plt

def csv(csv_file):
    pd_films = pd.read_csv(csv_file,sep = ',',header = 0)
    pd_films_sin_nan = pd_films.dropna(how='any')
    print pd_films_sin_nan
    pd_films_numeric = pd_films_sin_nan.select_dtypes(include=['float64','int64'])
    return pd_films_numeric

def analisis_exploratorio(pd_films_numeric):
    columns = pd_films_numeric.columns
    rows = pd_films_numeric.index
    score = columns[13]
    i = 1
    j = 1
    for column in columns:
        plt.figure(i)
        pd_films_numeric[column].plot(kind="hist")
        plt.show()
        i+=1
    
    for column in columns:
        plt.figure(j)
        pd_films_numeric.plot(kind = 'scatter',x= column,y = 'imdb_score')
        plt.show()
        j+=1
        
    scatter_matrix(pd_films_numeric, alpha = 0.2, figsize = (16,16), diagonal = 'kde')
    pd_films_numeric.corr()
    pd_films_numeric =pd.get_dummies(pd_films_numeric)
    scatter_matrix(pd_films_numeric, alpha = 0.2, figsize = (16,16), diagonal = 'kde')
    pd_films_numeric.plot(kind = 'scatter',x='imdb_score',y ='budget',z ='gross')
    
        
        
if __name__ == '__main__':
    matriz = csv('movie_metadata.csv')
    analisis_exploratorio(matriz)