import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.config import oot_years
from mmml.game_results import *

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def fnSplitData(data_all, save=False):
    """
    :param model_data: DataFrame.
    :param save: bool, default=False.
    :return: DataFrame, DataFrame
    """
    logging.info("Splitting data into Development and OOT sets...")

    # GET LIST OF YEARS TO USE AS DEV
    try:
        dev_years = list(set(data_all['Season'].unique()) - set(oot_years))
    except:
        dev_years = list(set([x[1] for x in data_all.index.get_values()]) - set(oot_years))

    dev_years.sort()
    logging.info("Dev seasons: {}".format(dev_years))
    data_dev = data_all.query('Season in {}'.format(dev_years))
    logging.info("Dev model data: {} obs".format(data_dev.shape[0]))

    # Save to Pickle
    data_dev.to_pickle('{}/Data/Processed/{}_dev.pkl'.format(os.path.dirname(os.getcwd()), save))
    # if save != False:
    #     model_data_dev.to_pickle('{}/Data/Processed/model_data_dev.pkl'.format(os.path.dirname(os.getcwd())))
    #     logging.info("Dev data saved to: {}/Data/Processed/model_data_dev.pkl".format(os.path.dirname(os.getcwd())))

    # GET LIST OF YEARS TO USE AS OOT
    oot_years.sort()
    logging.info("OOT seasons: {}".format(oot_years))
    data_oot = data_all.query('Season in {}'.format(oot_years))
    logging.info("OOT model data: {} obs".format(data_oot.shape[0]))

    # Save to Pickle
    data_oot.to_pickle('{}/Data/Processed/{}_oot.pkl'.format(os.path.dirname(os.getcwd()), save))
    # if save != False:
    #     model_data_oot.to_pickle('{}/Data/Processed/model_data_oot.pkl'.format(os.path.dirname(os.getcwd())))
    #     logging.info("OOT data saved to: {}/Data/Processed/model_data_oot.pkl".format(os.path.dirname(os.getcwd())))

    return data_dev, data_oot
