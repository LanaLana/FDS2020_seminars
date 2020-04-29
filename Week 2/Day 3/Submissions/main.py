import argparse
from loader import flights, build_dataset
from models import Classifiers

def set_args(parser):
    parser.add_argument('--url', default="https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz", \
                        type=str, help="URL to download dataset")
    parser.add_argument('--data_dir', default='data', type=str, help='Directory to store dataset')
    parser.add_argument('--rows_num', default=10000, type=int, help='Number of rows to use')
    return parser

def main():
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    
    flights(args.url, args.data_dir, args.rows_num)
    
    X_train, y_train, X_test, y_test = build_dataset(args.data_dir)
    
    for model_name in ["xgboost", "dask_xgboost"]: # "dask_xgboost"
        clf = Classifiers(model_name, (X_train, y_train, X_test, y_test))
        clf.run_clf()
        
if __name__ == '__main__':
    main()
    
