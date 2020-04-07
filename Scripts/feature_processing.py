import pandas as pd
import sys
import os
import logging
from sklearn import preprocessing
import pickle

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.game_results import *

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def fnEngineerFeatures(model_data, scaler=None, save=False):
    """

    :param model_data: DataFrame.
    :param scaler: sklearn.preprocessing.MinMaxScaler, default=None.
    :param save: bool, default=False.
    :return: DataFrame, dict, sklearn.preprocessing.MinMaxScaler
    """

    # ToDo - Allow ability to read in path to Pickle

    # Read Feature Data Dictionary
    # ToDo - clean up and remove the 2
    feature_list = pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(os.path.dirname(os.getcwd())))

    diff_cols = list(feature_list.query('Diff_Calc == True')['Name'])
    scale_cols = list(feature_list.query('Scale_Avg == True')['Name'])

    model_data = model_data.copy()
    logging.info("Creating target column...")
    model_data['HScore_diff'] = model_data['HScore'] - model_data['AScore']

    # Create H - A Difference Columns
    logging.info("Creating diff columns...")

    ### Check if columns are present
    for col in diff_cols:
        col = col.replace('_diff', '')
        if col+'_H' not in list(model_data.columns):
            print('throw error') # ToDo - Update to actually throw errors

        if col+'_A' not in list(model_data.columns):
            print('throw error') # ToDo - Update to actually throw errors

        model_data[col+'_diff'] = model_data[col+'_H'] - model_data[col+'_A']

    # Define Massey Columns to Scale
    logging.info("Scaling Massey Rankings...")
    scale_cols = list(feature_list.query('Scale_Avg == True')['Name'])

    # Fit Scaler
    if scaler is None:
        logging.info("Fitting Min-Max Scaler")
        min_max_scaler = preprocessing.MinMaxScaler()
        fitted_scaler = min_max_scaler.fit(pd.DataFrame(model_data[scale_cols]))

        # Save Min-Max Scaler
        with open("{}/Model_Objects/fitted_scaler.pkl".format(os.path.dirname(os.getcwd())), 'wb') as file:
            pickle.dump(fitted_scaler, file)
    else:
        # Handle Scaler
        if type(scaler) == preprocessing.data.MinMaxScaler:
            logging.info("Using Min-Max Scaler passed as argument")
            fitted_scaler = scaler
        else: # Handle path to scaler
            logging.info("Reading Min-Max Scaler from {}".format(scaler))
            with open(scaler, 'rb') as file:
                fitted_scaler = pickle.load(file)

    # DataFrame of Massey scaled columns
    scaled_df = pd.DataFrame(fitted_scaler.transform(model_data[scale_cols]),
        columns=[x+"_scaled" for x in scale_cols], index=model_data.index)

    logging.info("Creating average ranking of Massey columns")
    avg_rank = pd.DataFrame(scaled_df.mean(axis=1), columns=['Avg_Rank_diff'])

    # Merge Massey Avg Rank to Model Data
    model_data = model_data.merge(avg_rank, left_index=True, right_index=True)

    model_ids = list(feature_list.query('Include==True and Type=="ID"')['Name'])
    model_target = list(feature_list.query('Include==True and Type=="Target"')['Name'])
    model_features = list(feature_list.query('Include==True and Type=="Feature"')['Name'])

    # CREATE DICTIONARY OF IMPORTANT COLUMNS AND PURPOSES
    columns_key = {}
    columns_key['target'] = model_target
    columns_key['features'] = model_features
    columns_key['ids'] = model_ids

    processed_model_data = model_data[model_ids + model_target + model_features]

    # Save Processed DF to Pickle
    if save != False:
        if type(save) != bool: # If Passed Path
            model_data.to_pickle('{}/Data/Processed/{}.pkl'.format(os.path.dirname(os.getcwd()), save))
            logging.info("Model data saved to: {}/Data/Processed/{}.pkl".format(os.path.dirname(os.getcwd()), save))
        else: # Default Path
            model_data.to_pickle('{}/Data/Processed/processed_model_data.pkl'.format(os.path.dirname(os.getcwd())))
            logging.info("Model data saved to: {}/Data/Processed/processed_model_data.pkl".format(os.path.dirname(os.getcwd())))

    return processed_model_data, columns_key, fitted_scaler
