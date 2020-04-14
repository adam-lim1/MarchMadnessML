import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import data_folder, oot_years
from mmml.game_results import *
from mmml.utils import *

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
        dev_years = list(set([x[1] for x in data_all.index.to_numpy()]) - set(oot_years))

    dev_years.sort()
    logging.info("Dev seasons: {}".format(dev_years))
    data_dev = data_all.query('Season in {}'.format(dev_years))
    logging.info("Dev model data: {} obs".format(data_dev.shape[0]))

    # Save to Pickle
    if save != False:
        saveResults(object=data_dev, dir='Data/Processed', file_name="{}_dev.pkl".format(save))

    # GET LIST OF YEARS TO USE AS OOT
    oot_years.sort()
    logging.info("OOT seasons: {}".format(oot_years))
    data_oot = data_all.query('Season in {}'.format(oot_years))
    logging.info("OOT model data: {} obs".format(data_oot.shape[0]))

    # Save to Pickle
    if save != False:
        saveResults(object=data_oot, dir='Data/Processed', file_name="{}_oot.pkl".format(save))

    return data_dev, data_oot
