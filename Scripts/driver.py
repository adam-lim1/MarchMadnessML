import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path

from data_prep import fnDataPrep
from split_data import fnSplitData
from feature_processing import fnEngineerFeatures
from train import fnTrain
from score import fnScore

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

logging.info("###################### ENTERING DATA PREP ######################")
model_data = fnDataPrep(base_data_path, save=True)

logging.info("###################### ENTERING TEST/TRAIN SPLIT ######################")
model_data_dev, model_data_oot = fnSplitData(model_data, save=True)

print(list(model_data_dev))
logging.info("###################### ENTERING FEATURE PROCESSING ######################")
processed_model_data_dev, column_dict, fitted_scaler = fnEngineerFeatures(model_data_dev, save='processed_model_data_dev')
processed_model_data_oot, _, _ = fnEngineerFeatures(model_data_oot, fitted_scaler, save='processed_model_data_oot')

logging.info("###################### ENTERING MODEL TRAINING ######################")
clf, mean, std = fnTrain(processed_model_data_dev, column_dict, seed=96, save='xgboost_regression')

logging.info("###################### ENTERING MODEL SCORING ######################")
#y_pred = fnScore(processed_model_data_oot, column_dict, clf, mean, std)

logging.info("done")
