import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
#import dask_ml.xgboost
import pandas as pd
import numpy as np
import dask.dataframe as dd
from scipy_utils import make_cluster
import time
import os
import csv

class Classifiers():
    def __init__(self, name, dataset):
        self.name = name 
        self.dataset = dataset
        
    def df2dd(self): #convert pandas dataframe to dask format
        X_train, y_train, X_test, y_test = self.dataset
        X_train_df = dd.from_pandas(X_train) 
        y_train_df = dd.from_pandas(y_train)
        X_test_df = dd.from_pandas(X_test)
        y_test_df = dd.from_pandas(y_test)
        
        return X_train_df, y_train_df, X_test_df, y_test_df
    
    def run_clf(self):
        if self.name == "xgboost":
            accuracy, training_time, grid_search_time = self.simple_model()
        elif self.name == "dask_xgboost":
            self.dataset = df2dd(self.dataset)
            accuracy, training_time, grid_search_time = self.dask_model()
        
        if "stat_file.csv" in os.listdir("./"):
            open_mode = 'a'
        else:
            open_mode = 'w'
        with open("stat_file.csv", open_mode, newline ='') as f:
            Writer = csv.writer(f)  
            Writer.writerow([self.name, "   error: " + str(round(accuracy, 3)), \
                             " training time: "+str(round(training_time), 3), \
                             " grid search time: " + str(round(grid_search_time), 3)])
                
    def dask_model(self):
        '''
        cluster = make_cluster()
        cluster

        client = Client(cluster)
        client
        
        X_train, y_train, X_test, y_test = self.dataset
        time_start = time.time()
        clf = dask_ml.xgboost.train(client, params, X_train, y_train)
        train_time_dif = time.time() - time_start
        predictions = dask_ml.xgboost.predict(client, clf, X_test)
        mse = mean_squared_error(predictions.compute(), y_test)

        return mse, train_time_dif, grid_time_dif
        '''
        
        cluster = make_cluster()
        cluster

        client = Client(cluster)
        client
        
        X_train, y_train, X_test, y_test = self.dataset
        
        grid_values = {'max_depth': [3, 4, 5], 'learning_rate':[0.1, 0.01, 0.05]}
        clf = dask_ml.xgboost()
        grid_clf = dask_ml.model_selection.GridSearchCV(clf, param_grid=grid_values)
        time_start = time.time()
        grid_clf.fit(X_train, y_train)
        grid_time_dif = time.time() - time_start
        
        time_start = time.time()
        clf = dask_ml.xgboost.train(client, params, X_train, y_train)
        train_time_dif = time.time() - time_start
        predictions = dask_ml.xgboost.predict(client, clf, X_test)
        mse = mean_absolute_error(predictions.compute(), y_test)

        return mse, train_time_dif, grid_time_dif
    
    def simple_model(self):
        X_train, y_train, X_test, y_test = self.dataset
        data_dmatrix = xgb.DMatrix(X_train, y_train)
        
        clf = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 10)
        
        # search parameters
        grid_values = {'max_depth': [3, 4, 5], 'learning_rate':[0.1, 0.01, 0.05]}
        grid_clf = GridSearchCV(clf, param_grid = grid_values,scoring="neg_mean_squared_error", cv=3)
        time_start = time.time()
        grid_clf.fit(X_train,y_train)
        grid_time_dif = time.time() - time_start
        
        #train model with best parameters
        time_start = time.time()
        best_clf = xgb.train(dtrain=data_dmatrix, params=grid_clf.best_params_)
        train_time_dif = time.time() - time_start
        
        #Predict values based on new parameters
        test_dmatrix = xgb.DMatrix(X_test, y_test)
        preds = best_clf.predict(test_dmatrix)
        mae = mean_absolute_error(y_test, preds)

        return mae, train_time_dif, grid_time_dif


    