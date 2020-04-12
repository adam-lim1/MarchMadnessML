import pandas as pd
import sys
import os
import logging
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.game_results import *
from mmml.utils import *

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def reverse_base(base):
    reverse_base = base.copy()
    reverse_base = reverse_base.rename(columns={'HTeamID': 'ATeamID_2',
                                                'ATeamID':'HTeamID_2',
                                                'HScore':'AScore_2',
                                                'AScore':'HScore_2',
                                                'Seed_H':'Seed_A_2',
                                                'Seed_A':'Seed_H_2'})

    reverse_base = reverse_base.rename(columns={'ATeamID_2': 'ATeamID',
                                                'HTeamID_2':'HTeamID',
                                                'AScore_2':'AScore',
                                                'HScore_2':'HScore',
                                                'Seed_A_2':'Seed_A',
                                                'Seed_H_2':'Seed_H'})
    reverse_base['HWin'] = 1 - reverse_base['HWin']

    return reverse_base[base.columns]

def fnTrain(base, x_features, seed=96, save=False):
    """

    :param model_data: DataFrame.
    :param column_dict: dict.
    :param seed: int, default=96.
    :param save: bool, default=False.
    :return: sklearn.model_selection._search.GridSearchCV, double, double
    """

    ## READ FEATURE DICT
    logging.info("Reading feature dictionary...")
    # feature_list = pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(os.path.dirname(os.getcwd())))
    # diff_cols = list(feature_list.query('Diff_Calc==True')['Name'])
    # model_target = list(feature_list.query('Type=="Target" and Include==True')['Name'])
    # model_features = list(feature_list.query('Type=="Feature" and Include==True')['Name'])
    # model_ids = list(feature_list.query('Type=="ID"')['Name'])
    #
    # columns_key = {}
    # columns_key['target'] = model_target # ToDo - error handle if more than one entry
    # columns_key['features'] = model_features
    # columns_key['ids'] = model_ids
    # columns_key['diff_cols'] = diff_cols

    columns_key = getFeatureDict(pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(os.path.dirname(os.getcwd()))))

    logging.info("Creating model data...")
    model_data = createModelData(base, x_features, columns_key)

    # Reverse H/A notations on base dataset
    # Attempting to create predicitons that are the equal regardless of H/A status
    base_reverse = reverse_base(base)
    model_data_reverse = createModelData(base_reverse, x_features, columns_key)

    model_data_all = model_data.append(model_data_reverse)


    logging.info("Defining hyperparam grid...")
    parameters = {'max_depth': [3, 4, 5],
    'learning_rate':[0.1],
    'n_estimators': [10, 100, 1000], #number of trees, change it to 1000 for better results
    'gamma':[0, 0.05, 0.1],
    'min_child_weight':[0, 2, 4],
    'seed': [seed]} # binary:logistic

    xgb_model = xgb.XGBRegressor()
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, cv=5, verbose=0, refit=True)

    # Define features to use
    logging.info("Training target: {}".format(columns_key['target']))
    logging.info("Using Features: {}".format(columns_key['features']))

    # Fit Model
    logging.info("Fitting model...")
    clf.fit(model_data_all[columns_key['features']], model_data_all[columns_key['target']])

    logging.info("Best estimator:")
    print(clf.best_estimator_)

    logging.info("Best score:")
    print(clf.best_score_)

    logging.info("Feature importance:")
    print(pd.DataFrame(columns_key['features'], columns=['feature'])\
    .merge(pd.DataFrame(clf.best_estimator_.feature_importances_), left_index=True, right_index=True)\
    .sort_values(0, ascending=False))

    # Define Normal Distribution Parameters
    model = {}
    model['clf'] = clf
    model['mean'] = model_data_all[columns_key['target']].mean().get_values()[0]
    model['std'] = model_data_all[columns_key['target']].std().get_values()[0]

    # Save to Pickle
    if save != False:
        if type(save) != bool:
            with open("{}/Model_Objects/{}.pkl".format(os.path.dirname(os.getcwd()), save), 'wb') as file:
                    pickle.dump(model, file)

            logging.info("Model data saved to: {}/Model_Objects/{}.pkl".format(os.path.dirname(os.getcwd()), save))
        else:
            with open("{}/Model_Objects/clf.pkl".format(os.path.dirname(os.getcwd())), 'wb') as file:
                    pickle.dump(model, file)

            logging.info("Model data saved to: {}/Model_Objects/clf.pkl".format(os.path.dirname(os.getcwd())))

    return model
