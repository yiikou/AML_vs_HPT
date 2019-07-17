
from __future__ import print_function, division

from azureml.core.run import Run

import argparse
import os
import numpy as np
from statistics import *
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import auc,accuracy_score,roc_auc_score,auc,roc_curve

from azureml.core import Run

run = Run.get_context()

def download_data(download_url,dataset_name):
    import urllib
    data_file = dataset_name+'.csv'
    urllib.request.urlretrieve(download_url, filename=data_file)


    return data_file

def load_data(download_url,split_random_seeds):
    filename = os.path.join(os.getcwd(),download_url)
    print(filename)
    import pandas as pd
    df = pd.read_csv(filename)
    x_df = df.drop(columns =['target'])
    y_df = df[['target']]

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=split_random_seeds)
    # flatten y_train to 1d array
    y_train.values.flatten()
    return x_train, x_test, y_train, y_test

def load_outer_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-random-seeds', type=int, dest = 'split_random_seeds', help='Seed')
    parser.add_argument('--data-download-url', type=str, dest = 'data_download_url', help='Url')
    parser.add_argument('--dataset-name', type=str, dest = 'dataset_name', help='dataset_name')
    parser.add_argument('--output_dir', type=str, dest = 'output_dir', help='output_dir')

    parser.add_argument('--random_state', type=int, dest = 'random_state', help='max_depth')
       
    aargs = parser.parse_args()
    return aargs

def load_hyperparameter(aargs):
    
    random_state=aargs.random_state
    
    #run.log_row(name = 'Hyper parameter',
    #    mtry=aargs.mtry
    #)
    
    #run.log('mtry',np.int(aargs.mtry))
    return random_state

def setup_model(hyper_param,run):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state = hyper_param)
    return clf

def main():
    
    aargs = load_outer_parameter()

    data_local_path= download_data(aargs.data_download_url,aargs.dataset_name)
    
    x_train, x_test, y_train, y_test = load_data(data_local_path,aargs.split_random_seeds)

    hyper_param = load_hyperparameter(aargs)
    #print(str(hyper_param))
    #run.log("hyper_param", str(hyper_param))

    clf = setup_model(hyper_param,run)
    
    cv = ShuffleSplit(n_splits=100, test_size=0.33, random_state=aargs.split_random_seeds)
    
    auc_weighted_all = cross_val_score(clf, 
                                       x_train, y_train.values.flatten(), 
                                       cv=cv,
                                       scoring=make_scorer(roc_auc_score, 
                                                           average='weighted',
                                                           needs_threshold=True)
                                      )  
    
    run.log(name='Median auc_weighted', value=median(auc_weighted_all))
       
    print('auc_weighted is : '+ str(mean(auc_weighted_all)))
    run.log("auc_weighted", mean(auc_weighted_all))
    
    final_model =clf.fit(x_train, y_train.values.flatten())
    
    os.makedirs(aargs.output_dir, exist_ok=True)
    model_path_name = os.path.join(aargs.output_dir, 'model.pkl')
    import pickle
    pickle.dump(final_model, open(model_path_name, 'wb'))
    run.complete()

if __name__ == '__main__':
    main()