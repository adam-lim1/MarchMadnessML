import pandas as pd
import sys
import os
import logging
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import data_folder, log_location
from mmml.game_results import *
from mmml.utils import *

logger = setupLogger(name=__name__, output_file=log_location)

def reverse_base(base):
    """
    HELPER FUNCTION
    Reverses H/A orientation for a base DataFrame of matchups. Employed to
    effictively double size of training set and produce more symmetrical predictions.
    """
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
    Trains XGBoost Regression on set of base matchups. Utilizes a CV grid search
    across possible hyperparameters.

    Regression predicts difference in score from the home and away teams in the
    matchup (Home Score - Away Score). Combined with the mean and std deviation
    from the training set, we can transform the predicted difference in score to
    a probability that the Home team will win.

    Output is a dict with the fitted clf, mean target value from training set,
    and standard deviation of target values from the training set.
    """
    base_path = os.path.dirname(os.getcwd())

    ## READ FEATURE DICT
    logger.info("Reading feature dictionary...")
    columns_key = getFeatureDict(pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(base_path)))

    logger.info("Creating model data...")
    model_data = createModelData(base, x_features, columns_key)

    # Reverse H/A notations on base dataset
    # Attempting to create predicitons that are the equal regardless of H/A status
    base_reverse = reverse_base(base)
    model_data_reverse = createModelData(base_reverse, x_features, columns_key)

    model_data_all = model_data.append(model_data_reverse)

    logger.info("Defining hyperparam grid...")
    parameters = {'max_depth': [3, 4, 5],
    'learning_rate':[0.1],
    'n_estimators': [10, 100, 1000], #number of trees, change it to 1000 for better results
    'gamma':[0, 0.05, 0.1],
    'min_child_weight':[0, 2, 4],
    'seed': [seed]} # binary:logistic

    xgb_model = xgb.XGBRegressor()
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, cv=5, verbose=0, refit=True)

    # Define features to use
    logger.info("Training target: {}".format(columns_key['target']))
    logger.info("Using Features: {}".format(columns_key['features']))

    # Fit Model
    logger.info("Fitting model...")
    clf.fit(model_data_all[columns_key['features']], model_data_all[columns_key['target']])

    logger.info("Best estimator:")
    logger.info("{}".format(clf.best_estimator_))

    logger.info("Best score:")
    logger.info("{}".format(clf.best_score_))

    logger.info("Feature importance:")
    feature_importance = pd.DataFrame(columns_key['features'], columns=['feature'])\
    .merge(pd.DataFrame(clf.best_estimator_.feature_importances_), left_index=True, right_index=True)\
    .sort_values(0, ascending=False)
    logger.info("{}".format(feature_importance))

    # Define Normal Distribution Parameters
    model = {}
    model['clf'] = clf
    model['mean'] = model_data_all[columns_key['target']].mean()
    model['std'] = model_data_all[columns_key['target']].std()

    # Save to Pickle
    if save != False:
        saveResults(object=model, dir='Model_Objects', file_name='{}.pkl'.format(save))

    return model
