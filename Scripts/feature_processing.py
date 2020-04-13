import pandas as pd
import sys
import os
import logging
from sklearn import preprocessing
import pickle

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.game_results import *
from mmml.utils import *

#logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def fnScaleFeatures(x_features, scaler=None, save=False):
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
    columns_key = getFeatureDict(feature_list)
    scale_cols = columns_key['scale_cols']
    scale_cols = list(set([x[:-2] for x in scale_cols])) # Remove _H / _A suffixes

    x_features = x_features.copy()

    # Fit Scaler
    if scaler is None:
        logging.info("Fitting Min-Max Scaler")
        min_max_scaler = preprocessing.MinMaxScaler()
        fitted_scaler = min_max_scaler.fit(pd.DataFrame(x_features[scale_cols]))
        # Save Min-Max Scaler
        with open("{}/Model_Objects/fitted_scaler.pkl".format(os.path.dirname(os.getcwd())), 'wb') as file:
            pickle.dump(fitted_scaler, file)
    else:
        logging.info("Using Min-Max Scaler passed as argument")
        fitted_scaler = scaler
        # ToDo - Accomodate Path
        # Handle Scaler
        # if type(scaler) == preprocessing.data.MinMaxScaler:
        #     logging.info("Using Min-Max Scaler passed as argument")
        #     fitted_scaler = scaler

    # Transform DF
    scaled_df = pd.DataFrame(fitted_scaler.transform(x_features[scale_cols]),
    columns=[x+"_scaled" for x in scale_cols], index=x_features.index)

    # Average of scaled columns
    logging.info("Creating average ranking of Massey columns")
    avg_rank = pd.DataFrame(scaled_df[[x+"_scaled" for x in scale_cols]].mean(axis=1), columns=['Avg_Rank'])

    scaled_x_features = x_features.merge(avg_rank, left_index=True, right_index=True)

    # Save to Pickle
    scaled_x_features.to_pickle('{}/Data/Processed/{}.pkl'.format(os.path.dirname(os.getcwd()), save))

    return scaled_x_features, fitted_scaler
