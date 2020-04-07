import pandas as pd
import sys
import os
import logging

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.game_results import *

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def fnDataPrep(base_data_path, save=False):
    """
    Create base dataset for modeling from: Historical tournament matchups, tournament seeds/slots,
    season-aggregated regular season results, and Massey rankings. Transforms data from W/L
    orientation to Home/Away.
    :param base_data_path: str. Base path to datasets
    :param save: bool, default=False. Indicate whether to cache created dataframe
    as Pickle
    :return: DataFrame
    """
    reg_results_df = pd.read_csv('{}/MRegularSeasonDetailedResults.csv'.format(base_data_path))
    season_results = gameResults(reg_results_df)

    logging.info("Aggregating regular season stats...")
    season_stats = season_results.getSeasonStats()

    logging.info("Calculating regular season ELO's...")
    elo = season_results.getElo()

    logging.info("Getting end of season Massey Rankings...")
    massey = pd.read_csv('{}/MMasseyOrdinals.csv'.format(base_data_path))
    massey_final = massey.query('RankingDayNum == 133').copy()
    massey_final = massey_final.set_index(['TeamID','Season']).query('SystemName in ["POM", "SAG", "MOR"]') # "LMC", "EBP"
    massey_final = massey_final.drop('RankingDayNum', axis=1)
    massey_final = massey_final.pivot(columns='SystemName')['OrdinalRank']

    logging.info("Merging independent features...")
    x_features = season_stats.merge(elo, left_index=True, right_index=True)\
                            .merge(massey_final, left_index=True, right_index=True)

    logging.info("Creating base of past tournament games...")
    t_results_df = pd.read_csv('{}/MNCAATourneyDetailedResults.csv'.format(base_data_path))
    t_results = gameResults(t_results_df)
    base = t_results.getBase()

    round_lookup = {134: 0, 135: 0, 136: 1, 137: 1, 138: 2, 139: 2, 143: 3,
            144: 3, 145: 4, 146: 4, 152: 5, 154: 6}
    base['GameRound'] = base['DayNum'].apply(lambda x: round_lookup[x])

    ## Seeds
    seeds = pd.read_csv('{}/MNCAATourneySeeds.csv'.format(base_data_path))
    t_seeds = seeds[['Season', 'TeamID', 'Seed']]
    t_seeds.set_index(['TeamID', 'Season'], inplace=True)

    ## Tournament Slots
    slots_simple = pd.read_csv('{}/MNCAATourneySeedRoundSlots.csv'.format(base_data_path))
    slots_simple.drop('EarlyDayNum', axis=1, inplace=True)
    slots_simple.drop('LateDayNum', axis=1, inplace=True)
    slots_simple = slots_simple.set_index(['Seed', 'GameRound'])

    logging.info("Merging tournament games base to independent features...")
    # model_data = base.merge(x_features, left_on=['HTeamID', 'Season'], right_index=True)\
    #                 .merge(x_features, left_on=['ATeamID', 'Season'], right_index=True, suffixes=['_H', '_A'])\
    #                 .merge(t_seeds, left_on=['HTeamID', 'Season'], right_index=True)\
    #                 .merge(t_seeds, left_on=['ATeamID', 'Season'], right_index=True, suffixes=['_H', '_A'])

    model_data = base.merge(t_seeds, left_on=['HTeamID', 'Season'], right_index=True)\
                    .merge(t_seeds, left_on=['ATeamID', 'Season'], right_index=True, suffixes=['_H', '_A'])\
                    .merge(slots_simple, left_on=['Seed_H', 'GameRound'], right_index=True)\
                    .merge(x_features, left_on=['HTeamID', 'Season'], right_index=True)\
                    .merge(x_features, left_on=['ATeamID', 'Season'], right_index=True, suffixes=['_H', '_A'])

    logging.info("Model data created")

    if save != False:
        model_data.to_pickle('{}/Data/Processed/model_data.pkl'.format(os.path.dirname(os.getcwd())))
        logging.info("Model data saved to: {}/Data/Processed/model_data.pkl".format(os.path.dirname(os.getcwd())))

    return model_data
