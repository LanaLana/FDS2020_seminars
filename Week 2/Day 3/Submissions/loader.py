from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob
from sklearn.model_selection import train_test_split


def flights(url = "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz", \
            data_dir='data', rows_num=10000):
    
    flights_raw = os.path.join(data_dir, 'nycflights.tar.gz')
    flightdir = os.path.join(data_dir, 'nycflights')
    jsondir = os.path.join(data_dir, 'flightjson')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print("- Downloading NYC Flights dataset... ", end='', flush=True)
        urllib.request.urlretrieve(url, flights_raw)
        print("done", flush=True)

    if not os.path.exists(flightdir):
        print("- Extracting flight data... ", end='', flush=True)
        tar_path = os.path.join(data_dir, 'nycflights.tar.gz')
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall(data_dir+'/')
        print("done", flush=True)

    if not os.path.exists(jsondir):
        print("- Creating json data... ", end='', flush=True)
        os.mkdir(jsondir)
        for path in glob(os.path.join(data_dir, 'nycflights', '*.csv')):
            prefix = os.path.splitext(os.path.basename(path))[0]
            # Just take the first 10000 rows for the demo
            df = pd.read_csv(path).iloc[:rows_num]
            df.to_json(os.path.join(data_dir, 'flightjson', prefix + '.json'),
                       orient='records', lines=True)
        print("done", flush=True)

    print("** Finished! **")

def build_dataset(data_dir='data', fill_nan=False, one_hot_encod=False):
    filenames = glob('./{}/nycflights/*.csv'.format(data_dir))
    dataframes = [pd.read_csv(f) for f in filenames]

    frame = pd.concat(dataframes, axis=0, ignore_index=True)
    
    if fill_nan:
        print("Not implemented") # should be written
        pass
    else:
        frame = frame.dropna()
      
    if one_hot_encod:
        print("Not implemented") # should be written
        pass
    else:
        frame = frame.select_dtypes(['number'])
        
    frame = frame.sample(frac=1)
    train, test = train_test_split(frame, test_size=0.2)

    # separate features and target
    X_train = train.drop(columns=['DepDelay'])
    X_test = test.drop(columns=['DepDelay'])

    y_train = train['DepDelay']
    y_test = test['DepDelay']
    
    return X_train, y_train, X_test, y_test
