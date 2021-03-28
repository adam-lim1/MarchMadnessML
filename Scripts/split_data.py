import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import data_folder, oot_years, log_location
from mmml.game_results import *
from mmml.utils import *

logger = setupLogger(name=__name__, output_file=log_location)

def fnSplitData(data_all, save=False):
    """
    Splits data into development and out of time sets based on oot_years value specified
    in config file.

    :param data_all: original DataFrame to split
    :param save: default=False. String value specifies file name to save split DF as
    :return: development DataFrame, OOT DataFrame
    """

    logger.info("Splitting data into Development and OOT sets...")

    # GET LIST OF YEARS TO USE AS DEV
    try:
        dev_years = list(set(data_all['Season'].unique()) - set(oot_years))
    except:
        dev_years = list(set([x[1] for x in data_all.index.to_numpy()]) - set(oot_years))
    dev_years = [x for x in dev_years if x != 2020] # Do not use 2020 data due to COVID-19 disruption

    dev_years.sort()
    logger.info("Dev seasons: {}".format(dev_years))
    data_dev = data_all.query('Season in {}'.format(dev_years))
    logger.info("Dev model data: {} obs".format(data_dev.shape[0]))

    # Save to Pickle
    if save != False:
        saveResults(object=data_dev, dir='Data/Processed', file_name="{}_dev.pkl".format(save))

    # GET LIST OF YEARS TO USE AS OOT
    oot_years.sort()
    logger.info("OOT seasons: {}".format(oot_years))
    data_oot = data_all.query('Season in {}'.format(oot_years))
    logger.info("OOT model data: {} obs".format(data_oot.shape[0]))

    # Save to Pickle
    if save != False:
        saveResults(object=data_oot, dir='Data/Processed', file_name="{}_oot.pkl".format(save))

    return data_dev, data_oot
