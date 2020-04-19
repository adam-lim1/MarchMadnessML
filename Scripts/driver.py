import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import log_location
from mmml.utils import *

from data_prep import fnDataPrep
from split_data import fnSplitData
from feature_processing import fnScaleFeatures
from train import fnTrain
from score import fnScore, fnEvaluate, fnGetBracket

logger = setupLogger(name=__name__, output_file=log_location)

logger.info("###################### ENTERING DATA PREP ######################")
x_features, base = fnDataPrep(save=True)

logger.info("###################### ENTERING TEST/TRAIN SPLIT ######################")
x_features_dev, x_features_oot = fnSplitData(x_features, save='x_features')
base_dev, base_oot = fnSplitData(base, save='base')

logger.info("###################### ENTERING FEATURE PROCESSING ######################")
scaled_x_features_dev, fitted_scaler = fnScaleFeatures(x_features_dev, save='scaled_x_features_dev')
scaled_x_features_oot, _ = fnScaleFeatures(x_features_oot, fitted_scaler, save='scaled_x_features_oot')

logger.info("###################### ENTERING MODEL TRAINING ######################")
model = fnTrain(base_dev, scaled_x_features_dev, seed=96, save='xgboost_regression_reverse')

logger.info("###################### ENTERING MODEL SCORING ######################")
_, results_df_chalk = fnScore(base_oot, scaled_x_features_oot, scorer='chalk', seed=69)
_, results_df_model = fnScore(base_oot, scaled_x_features_oot, scorer=model)

logger.info("Evaluating chalk predictions: (Year: Overall Accuracy, ESPN Bracket Pts)")
_ = fnEvaluate(results_df_chalk)

logger.info("Evaluating model predictions: (Year: Overall Accuracy, ESPN Bracket Pts)")
_ = fnEvaluate(results_df_model)

logger.info("Rendering full bracket predictions...")
bracket = fnGetBracket(results_df_model.query('Season==2019'), save='2019_bracket_predictions')
logger.info("{}".format(bracket.head()))

logger.info("done")
