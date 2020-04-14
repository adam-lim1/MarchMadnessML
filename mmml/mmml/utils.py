import pandas as pd
import logging
import os
import pickle

def saveResults(object, dir, file_name):
    base_path = os.path.dirname(os.getcwd())

    print(os.path.join(base_path, dir, file_name))

    # Create directory if doesn't exist
    if not os.path.exists(os.path.join(base_path, dir)):
        logging.info("Creating directory: {}".format(os.path.join(base_path, dir)))
        os.makedirs(os.path.join(base_path, dir))

    if isinstance(object, pd.DataFrame):
        logging.info("Writing object to Pickle: {}".format(os.path.join(base_path, dir, file_name)))
        object.to_pickle(os.path.join(base_path, dir, file_name))
    else:
        logging.info("Writing object to Pickle: {}".format(os.path.join(base_path, dir, file_name)))
        with open(os.path.join(base_path, dir, file_name), 'wb') as file:
            pickle.dump(object, file)

    return 1

def getFeatureDict(feature_list):
    #feature_list = pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(os.path.dirname(os.getcwd())))
    diff_cols = list(feature_list.query('Diff_Calc==True')['Name'])
    scale_cols = list(feature_list.query('Scale_Avg == True')['Name'])
    model_target = list(feature_list.query('Type=="Target" and Include==True')['Name'])
    model_features = list(feature_list.query('Type=="Feature" and Include==True')['Name'])
    model_ids = list(feature_list.query('Type=="ID" and Include==True')['Name'])

    columns_key = {}
    columns_key['target'] = model_target # ToDo - error handle if more than one entry
    columns_key['features'] = model_features
    columns_key['ids'] = model_ids
    columns_key['diff_cols'] = diff_cols
    columns_key['scale_cols'] = scale_cols

    return columns_key

def createModelData(base, x_features, columns_key):
    ##### 1. Merge Base and X Features
    logging.info("Merging base and x features...")
    model_data = base.merge(x_features, left_on=['HTeamID', 'Season'], right_index=True)\
                    .merge(x_features, left_on=['ATeamID', 'Season'], right_index=True, suffixes=['_H', '_A'])

    ## Create Diff Cols
    logging.info("Creating diff cols...")
    for x in columns_key['diff_cols']:
        model_data[x] = model_data[x.replace('_diff', '_H')] - model_data[x.replace('_diff', '_A')]

    ## Create Target #ToDo - Make this not hardcoded
    logging.info("Creating target...")
    try:
        model_data['HScore_diff'] = model_data['HScore'] - model_data['AScore']
    except:
        model_data['HScore_diff'] = -999

    return model_data[columns_key['ids'] + columns_key['target'] + columns_key['features']].copy()
