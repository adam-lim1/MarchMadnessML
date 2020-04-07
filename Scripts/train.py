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

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def fnTrain(model_data, column_dict, seed=96, save=False):
    """

    :param model_data: DataFrame.
    :param column_dict: dict.
    :param seed: int, default=96.
    :param save: bool, default=False.
    :return: sklearn.model_selection._search.GridSearchCV, double, double
    """

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
    included_features = column_dict['features']
    target = column_dict['target']
    logging.info("Training target: {}".format(target))
    logging.info("Using Features: {}".format(included_features))

    # Fit Model
    logging.info("Fitting model...")
    clf.fit(model_data[included_features], model_data[target])

    logging.info("Best estimator:")
    print(clf.best_estimator_)

    logging.info("Feature importance:")
    print(pd.DataFrame(included_features, columns=['feature'])\
    .merge(pd.DataFrame(clf.best_estimator_.feature_importances_), left_index=True, right_index=True)\
    .sort_values(0, ascending=False))

    # Define Normal Distribution Parameters
    mean = model_data[target].mean()
    std = model_data[target].std()

    # Save to Pickle
    if save != False:
        if type(save) != bool:
            with open("{}/Model_Objects/{}.pkl".format(os.path.dirname(os.getcwd()), save), 'wb') as file:
                    pickle.dump(clf, file)

            logging.info("Model data saved to: {}/Model_Objects/{}.pkl".format(os.path.dirname(os.getcwd()), save))
        else:
            with open("{}/Model_Objects/clf.pkl".format(os.path.dirname(os.getcwd())), 'wb') as file:
                    pickle.dump(clf, file)

            logging.info("Model data saved to: {}/Model_Objects/clf.pkl".format(os.path.dirname(os.getcwd())))

    return clf, mean, std
