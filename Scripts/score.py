import pandas as pd
import sys
import os
import logging
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle
from scipy.stats import norm

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import data_folder, log_location, current_season
from mmml.game_results import *
from mmml.utils import *

logger = setupLogger(name=__name__, output_file=log_location)

### HELPERS FOR CHALK PREDICTION ALG
def getNumericSeed(seed):
    """
    HELPER FUNCTION
    Convert Region/Seed into strictly numeric seed value
    Example: 'W01' -> 1
    """
    seed = seed[1:]

    if seed[-1] in ['a', 'b']:
        seed = seed[:-1]

    return int(seed)

def chalk_predictions(df, seed=42):
    """
    HELPER FUNCTION
    Baseline prediciton algorithm. Predict higher seeds will win every time.
    If both teams seeds are equal, choose winner based on random value

    :output: DF with Pred and Prob columns, indexed by input DF index.
    """
    score_df = df.copy()

    # Add Random for if
    np.random.seed(seed=seed)
    score_df['rand'] = np.random.rand(score_df.shape[0])

    # Get Numeric Seeds
    score_df['Seed_H_Numeric'] = score_df['Seed_H'].apply(lambda x: getNumericSeed(x))
    score_df['Seed_A_Numeric'] = score_df['Seed_A'].apply(lambda x: getNumericSeed(x))

    score_df['Pred'] = np.where(score_df['Seed_H_Numeric'] < score_df['Seed_A_Numeric'], 1,
                                np.where(score_df['Seed_H_Numeric'] > score_df['Seed_A_Numeric'], 0,
                                        np.where(score_df['rand'] < 0.5, 1, 0)))
    score_df['Prob'] = np.where(score_df['Pred'] == 1, 1.0, 0.0)

    prediction_df = score_df[['Pred','Prob']].set_index(score_df.index)
    return prediction_df

def model_predictions(df, model, features):
    """
    HELPER FUNCTION
    Score dataframe with fitted sklearn classifier.

    :param df: DF (combined base and x_features) to score
    :param model: dict of clf, mean target, std target to score/transform df with
    :param features: list of features to pass to classifier
    :output: DF with Pred and Prob columns, indexed by input DF index
    """
    # Produce Scores
    y_pred = model['clf'].predict(df[features])
    y_pred = pd.DataFrame(y_pred, columns=['Pred'], index=df.index)

    y_pred['Prob'] = y_pred['Pred'].apply(lambda x: norm(model['mean'], model['std']).cdf(x))

    y_pred['Pred'] = np.where(y_pred['Pred'] > 0, 1, 0)

    return y_pred

def score_round(matchups_r1, x_features, columns_key, scorer='chalk', seed=42):
    """
    HELPER FUNCTION
    Sub-function to score matchups for round n and produce matchups for round
    n+1 based on predicted winners from n. Allows for a scoring assessment
    replicating better real-life process (rather than straight accuracy on known
    games).

    :param matchups_r1: DF of H/A matchups for tourmanent round
    :param x_features: DF of x_features
    :param columns_key: Dict containing feature columns to pass to classifier
    :param scorer: default = 'chalk'. Scoring algorithm to use. chalk=chalk_predictions scorer
    Otherwise, must pass a model dict containing clf, mean of target, std of target
    :param seed: Seed to use in chalk scoring function
    :output: DF with base matchups for round n+1, DF with predicted outcomes for round n
    """
    base_path = os.path.dirname(os.getcwd())
    # Merge Current Round Matchups with X_Features
    score_r1 = createModelData(base=matchups_r1, x_features=x_features, columns_key=columns_key)

    # Score round matchups
    if scorer=='chalk':
        predictions = chalk_predictions(df=score_r1, seed=seed)
    else:
        predictions = model_predictions(df=score_r1, model=scorer, features=columns_key['features']) # ToDo - pass features?

    # Merge Predictions/Probabilites back to Score DF
    pred_r1 = score_r1.merge(predictions, left_index=True, right_index=True)

    # Identify Seed/TeamID of predicted Winner
    pred_r1['WTeamID_pred'] = np.where(pred_r1['Pred']==1, pred_r1['HTeamID'], pred_r1['ATeamID'])
    pred_r1['WSeed_pred'] = np.where(pred_r1['Pred']==1, pred_r1['Seed_H'], pred_r1['Seed_A'])

    # Get Slots
    slots_simple = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneySeedRoundSlots.csv'.format(base_path, data_folder))
    slots_simple.drop('EarlyDayNum', axis=1, inplace=True)
    slots_simple.drop('LateDayNum', axis=1, inplace=True)
    slots_simple = slots_simple.set_index(['Seed', 'GameRound'])

    # Combine Next Round Slots and Current Round Predictions
    r2_slots = pred_r1[['Season', 'GameRound', 'WSeed_pred', 'WTeamID_pred']].copy()
    #r2_slots['GameRound'] = round_nbr + 1
    r2_slots['GameRound'] = r2_slots['GameRound'] + 1
    r2_slots = r2_slots.merge(slots_simple, left_on=['WSeed_pred', 'GameRound'], right_index=True, how='left')
    r2_slots = r2_slots.rename(columns={'WSeed_pred':'Seed', 'WTeamID_pred':'TeamID'})

    # Create H/A Team Matchups based on GameSlot
    r2_slots.sort_values(['Season', 'GameSlot', 'Seed'], inplace=True)
    r2_slots_H = r2_slots.drop_duplicates(subset=['Season','GameSlot'], keep='first')
    r2_slots_A = r2_slots.drop_duplicates(subset=['Season','GameSlot'], keep='last')

    base_r2 = r2_slots_H.merge(r2_slots_A,
                      left_on=['Season', 'GameRound', 'GameSlot'],
                      right_on=['Season', 'GameRound', 'GameSlot'],
                     how='inner', suffixes=['_H', '_A'])

    base_r2 = base_r2.rename(columns={'TeamID_H':'HTeamID', 'TeamID_A':'ATeamID'})

    return base_r2, pred_r1

def fnScore(base, x_features, scorer='chalk', seed=42):
    """
    Produces true tournament predictions where the preditions for each round are
    conditional on the predictions from the prior round.

    Chalk scorer produces baseline score if were to pick the higher seed every game.
    Fitted models can be evaluated by passing a model dict to the scoring input.

    Output is a dict with each round's matchups and predictions and a DF with all
    round predictions combined together.

    This will not create scores for the current_season
    """
    base_path = os.path.dirname(os.getcwd())

    # ToDo - Split for simple scoring method?

    ## READ FEATURE DICT
    columns_key = getFeatureDict(pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(base_path)))

    #### Get Initial Set of True Results
    true_outcome = base.copy()
    true_outcome['WTeamID_true'] = np.where(true_outcome['HWin']==1, true_outcome['HTeamID'], true_outcome['ATeamID'])
    true_outcome = true_outcome[['Season', 'GameRound', 'GameSlot', 'WTeamID_true']]

    #### Create dictionary of base matchups and predictions for each round
    round_dict = {i:{'base':None,'pred':None} for i in range(1,7)}
    round_dict[1]={'base':base.query('GameRound==1')}

    #### Score each round and create next round matchups
    for round_num in range(1,7):
        logger.info("Getting predictions for Round {}...".format(round_num))
        base_r, pred_r0 = score_round(round_dict[round_num]['base'], x_features, columns_key, scorer=scorer, seed=seed)

        round_dict[round_num]['pred'] = pred_r0

        if round_num != 6:
            round_dict[round_num+1]['base'] = base_r

    #### CREATE OVERALL RESULTS DF
    logger.info("Creating DataFrame of all results...")
    results_df = round_dict[1]['pred']

    for i in range(2,7):
        results_df = results_df.append(round_dict[i]['pred'])

    results_df = results_df.merge(true_outcome, left_on=['Season', 'GameRound', 'GameSlot'], right_on=['Season', 'GameRound', 'GameSlot'], how='left')

    #### SCORE VS TRUE RESULTS
    # logging.info("Scoring...")

    # Flag if prediction correct
    # results_df['Correct'] = np.where(results_df['WTeamID_true'] == results_df['WTeamID_pred'], 1, 0)

    # Convert results to ESPN type score
    # points_dict = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}
    # results_df['Points'] = np.where(results_df['Correct']==1, results_df['GameRound'].apply(lambda x: points_dict[x]), 0)

    return round_dict, results_df

def fnEvaluate(results_df):
    """
    Given DF of predictions of historical games, calculate overall accuracy and
    ESPN style bracket points, printing results both overall and by round
    """
    # Create Flag if prediction correct
    results_df['Correct'] = np.where(results_df['WTeamID_true'] == results_df['WTeamID_pred'], 1, 0)

    # GameRound: Point value from ESPN bracket challenge
    points_dict = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}
    results_df['Points'] = np.where(results_df['Correct']==1, results_df['GameRound'].apply(lambda x: points_dict[x]), 0)

    year_list = list(set(results_df['Season']))
    year_list.sort()

    # Results for each year
    results_dict = {}
    for year in year_list: # for year in year_list:
        acc = results_df.query('Season=={}'.format(year))['Correct'].sum() / results_df.query('Season=={}'.format(year))['Correct'].count()
        pts = results_df.query('Season=={}'.format(year))['Points'].sum()

        by_round_correct = results_df.query('Season=={}'.format(year)).groupby('GameRound').sum()['Correct']
        by_round_total = results_df.query('Season=={}'.format(year)).groupby('GameRound').count()['Correct']
        by_round_pts = results_df.query('Season=={}'.format(year)).groupby('GameRound').sum()['Points']
        by_round_results = pd.DataFrame(by_round_pts).merge(pd.DataFrame(by_round_correct / by_round_total), left_index=True, right_index=True).transpose()

        # print("")
        logger.info("{year}: {acc}, {pts}".format(year=year, acc=acc, pts=pts))
        logger.info("\n {} \n".format(by_round_results))
        results_dict[year] = {'acc':acc, 'pts':pts}
    return results_dict

def fnGetBracket(results_df, save=False):
    """
    Given DF of predictions, produce CSV of matchups and predictions
    """
    base_path = os.path.dirname(os.getcwd())

    # Append Team name information
    teams = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MTeams.csv'.format(base_path, data_folder))
    merged = results_df.merge(teams, left_on='HTeamID', right_on='TeamID')\
                        .merge(teams, left_on='ATeamID', right_on='TeamID', suffixes=['_H', '_A'])

    bracket = merged[['Season', 'GameRound', 'Seed_H', 'Seed_A', 'TeamName_H', 'TeamName_A', 'Pred', 'Prob']].copy()

    bracket.sort_values(by=['Season', 'GameRound', 'Seed_H'], inplace=True)

    # Save to CSV
    if save != False:
        if not os.path.exists('{}/Output'.format(base_path)):
            os.makedirs('{}/Output'.format(base_path))

        bracket.to_csv('{}/Output/{}.csv'.format(base_path, save))

    return bracket

def predictCurrentYear(base, x_features, scorer='chalk', seed=42):
    """
    Create bracket predictions for current season

    Similar to fnScore but does not require true game outcomes
    """
    base_path = os.path.dirname(os.getcwd())

    ## READ FEATURE DICT
    columns_key = getFeatureDict(pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(base_path)))

    #### 2021 Workaround: Append first round of games to base
    slots = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneySlots.csv'.format(base_path, data_folder))
    round_slots = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneySeedRoundSlots.csv'.format(base_path, data_folder))
    seeds = pd.read_csv('{}/Data/Raw/{}/MDataFiles_Stage1/MNCAATourneySeeds.csv'.format(base_path, data_folder))

    seeds_current = seeds.query('Season=={}'.format(current_season))
    round1_current = slots[slots['Slot'].apply(lambda x: x[0:2] == "R1")==True].query('Season=={}'.format(current_season)) # Identify first round
    
    # Temp work around - choose default winner for play in games
    round1_current = round1_current\
                        .replace('W11','W11a')\
                        .replace('W16','W16a')\
                        .replace('X11','X11a')\
                        .replace('X16','X16a')
    # Append team ID's
    round1_current = round1_current.merge(seeds_current, left_on=['StrongSeed'], right_on=['Seed']).rename(columns={'TeamID':'HTeamID'}).drop('Seed',axis=1)
    round1_current = round1_current.merge(seeds_current, left_on=['WeakSeed'], right_on=['Seed'], how='left').rename(columns={'TeamID':'ATeamID'}).drop('Seed',axis=1)
    round1_current = round1_current.rename(columns={'StrongSeed':'Seed_H','WeakSeed':'Seed_A', 'Slot':'GameSlot'})

    round1_current['GameRound'] = 1
    round1_current['DayNum'] = 999
    round1_current['HWin'] = 999
    round1_current['HScore'] = 999
    round1_current['AScore'] = 999
    round1_current_format = round1_current[list(base.columns)]
     
    # print(round1_current_format)
    base = round1_current_format.copy()

    #### Create dictionary of base matchups and predictions for each round
    round_dict = {i:{'base':None,'pred':None} for i in range(1,7)}
    round_dict[1]={'base':base.query('GameRound==1')}

    #### Score each round and create next round matchups
    for round_num in range(1,7):
        logger.info("Getting predictions for Round {}...".format(round_num))
        base_r, pred_r0 = score_round(round_dict[round_num]['base'], x_features, columns_key, scorer=scorer, seed=seed)

        round_dict[round_num]['pred'] = pred_r0

        if round_num != 6:
            round_dict[round_num+1]['base'] = base_r

    #### CREATE OVERALL RESULTS DF
    logger.info("Creating DataFrame of all results...")
    results_df = round_dict[1]['pred']

    for i in range(2,7):
        results_df = results_df.append(round_dict[i]['pred'])

    print(results_df)

    return round_dict, results_df
