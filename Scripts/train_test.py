import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.config import oot_years
from mmml.game_results import *

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def fnSplitData(model_data, save=False):
    logging.info("Splitting data into Development and OOT sets...")

    dev_years = list(set(list(model_data['Season'].unique())) - set(oot_years))
    dev_years.sort()
    logging.info("Dev seasons: {}".format(dev_years))
    model_data_dev = model_data.query('Season in {}'.format(dev_years))
    logging.info("Dev model data: {} obs".format(model_data_dev.shape[0]))

    if save != False:
        model_data_dev.to_pickle('{}/Data/Processed/model_data_dev.pkl'.format(os.path.dirname(os.getcwd())))
        logging.info("Dev data saved to: {}/Data/Processed/model_data_dev.pkl".format(os.path.dirname(os.getcwd())))

    oot_years.sort()
    logging.info("OOT seasons: {}".format(oot_years))
    model_data_oot = model_data.query('Season in {}'.format(oot_years))
    logging.info("OOT model data: {} obs".format(model_data_oot.shape[0]))

    if save != False:
        model_data_oot.to_pickle('{}/Data/Processed/model_data_oot.pkl'.format(os.path.dirname(os.getcwd())))
        logging.info("OOT data saved to: {}/Data/Processed/model_data_oot.pkl".format(os.path.dirname(os.getcwd())))

    return model_data_dev, model_data_oot
