import pandas as pd
import logging
import os
import pickle

def setupLogger(name=__name__, level=logging.INFO, output_file=None):
    # Initiate Logger
    logger = logging.getLogger(name)

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s - %(message)s')
    # fhandler.setFormatter(formatter)

    # Set Handlers
    shandler = logging.StreamHandler()
    logger.addHandler(shandler)
    shandler.setFormatter(formatter)

    # If output_file is None, log to stream only; otherwise log to file
    if output_file is not None:
        fhandler = logging.FileHandler(filename=output_file, mode='a')
        logger.addHandler(fhandler)
        fhandler.setFormatter(formatter)

    return logger


def saveResults(object, dir, file_name):
    """
    Save results of object as pickle, creating dir if doesn't already exist
    """
    base_path = os.path.dirname(os.getcwd())

    print(os.path.join(base_path, dir, file_name))

    # Create directory if doesn't exist
    if not os.path.exists(os.path.join(base_path, dir)):
        # logging.info("Creating directory: {}".format(os.path.join(base_path, dir)))
        os.makedirs(os.path.join(base_path, dir))

    if isinstance(object, pd.DataFrame): # If DataFrame
        # logging.info("Writing object to Pickle: {}".format(os.path.join(base_path, dir, file_name)))
        object.to_pickle(os.path.join(base_path, dir, file_name))
    else: # If Dict/model object
        # logging.info("Writing object to Pickle: {}".format(os.path.join(base_path, dir, file_name)))
        with open(os.path.join(base_path, dir, file_name), 'wb') as file:
            pickle.dump(object, file)

    return 1

def getFeatureDict(feature_list):
    """
    Given DF of feature information, parse into dictionary
    """
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
    """
    Helper function to merge a base of game matchups and historical x-features
    for each team. Creates diff columns between _H/_A features. Also creates
    target of H - A score differential.

    Returns df of ID columns, target column, and feature columns
    """

    ##### 1. Merge Base and X Features
    # logging.info("Merging base and x features...")
    model_data = base.merge(x_features, left_on=['HTeamID', 'Season'], right_index=True)\
                    .merge(x_features, left_on=['ATeamID', 'Season'], right_index=True, suffixes=['_H', '_A'])

    ## Create Diff Cols
    # logging.info("Creating diff cols...")
    for x in columns_key['diff_cols']:
        model_data[x] = model_data[x.replace('_diff', '_H')] - model_data[x.replace('_diff', '_A')]

    ## Create Target #ToDo - Make this not hardcoded
    # logging.info("Creating target...")
    try:
        model_data['HScore_diff'] = model_data['HScore'] - model_data['AScore']
    except:
        model_data['HScore_diff'] = -999

    return model_data[columns_key['ids'] + columns_key['target'] + columns_key['features']].copy()
