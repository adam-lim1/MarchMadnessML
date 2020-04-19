import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import data_folder, log_location
from mmml.game_results import *
from mmml.utils import *

logger = setupLogger(name=__name__, output_file=log_location)

def fnDataPrep(save=False):
    """
    Create base datsets for modeling. Creates a dataset of X (independent) features
    and a base dataset of historical tournament matchups.

    For X features, calculates and aggregates regular season aggregate stats,
    ELO, Massey rankings. For Base matchups, appends seed and tournament slot info.
    Transforms W/L format to Home/Away.

    :param save: bool, default=False. Indicate whether to cache base and x_features dataframes
    :return: DataFrame of X features, DataFrame of base matchups
    """

    base_path = os.path.dirname(os.getcwd())

    reg_results_df = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv'.format(base_path, data_folder))
    season_results = gameResults(reg_results_df)

    logger.info("Aggregating regular season stats...")
    season_stats = season_results.getSeasonStats()

    logger.info("Calculating regular season ELO's...")
    elo = season_results.getElo()

    logger.info("Getting end of season Massey Rankings...")
    massey = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MMasseyOrdinals.csv'.format(base_path, data_folder))
    massey_final = massey.query('RankingDayNum == 133').copy()
    massey_final = massey_final.set_index(['TeamID','Season']).query('SystemName in ["POM", "SAG", "MOR"]') # "LMC", "EBP"
    massey_final = massey_final.drop('RankingDayNum', axis=1)
    massey_final = massey_final.pivot(columns='SystemName')['OrdinalRank']

    logger.info("Merging independent features...")
    x_features = season_stats.merge(elo, left_index=True, right_index=True)\
                            .merge(massey_final, left_index=True, right_index=True)

    # ToDo - Improve saving process
    if save != False:
        saveResults(object=x_features, dir='Data/Processed/', file_name='x_features.pkl')

    ## Part 2 - Create Base
    logger.info("Creating base of past tournament games...")
    t_results_df = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv'.format(base_path, data_folder))
    t_results = gameResults(t_results_df)
    base = t_results.getBase()

    round_lookup = {134: 0, 135: 0, 136: 1, 137: 1, 138: 2, 139: 2, 143: 3,
            144: 3, 145: 4, 146: 4, 152: 5, 154: 6}
    base['GameRound'] = base['DayNum'].apply(lambda x: round_lookup[x])

    ## Seeds
    seeds = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneySeeds.csv'.format(base_path, data_folder))
    t_seeds = seeds[['Season', 'TeamID', 'Seed']]
    t_seeds.set_index(['TeamID', 'Season'], inplace=True)

    ## Tournament Slots
    slots_simple = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneySeedRoundSlots.csv'.format(base_path, data_folder))
    slots_simple.drop('EarlyDayNum', axis=1, inplace=True)
    slots_simple.drop('LateDayNum', axis=1, inplace=True)
    slots_simple = slots_simple.set_index(['Seed', 'GameRound'])

    base = base.merge(t_seeds, left_on=['HTeamID', 'Season'], right_index=True, how='left')\
                .merge(t_seeds, left_on=['ATeamID', 'Season'], right_index=True, how='left', suffixes=['_H', '_A'])\
                .merge(slots_simple, left_on=['Seed_H', 'GameRound'], right_index=True)

    # ToDo - Improve saving process
    if save != False:
        saveResults(object=base, dir='Data/Processed', file_name='base.pkl')

    return x_features, base
