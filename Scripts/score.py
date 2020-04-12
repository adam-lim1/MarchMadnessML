import pandas as pd
import sys
import os
import logging
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import pickle
from scipy.stats import norm

sys.path.append('{}/mmml'.format(os.path.dirname(os.getcwd())))
from mmml.config import base_data_path
from mmml.game_results import *
from mmml.utils import *

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

### HELPERS FOR CHALK PREDICTION ALG
def getNumericSeed(seed):
    """
    Helper function for convering Region/Seed into strictly numeric value
    Example: 'W01' -> 1
    """
    seed = seed[1:]

    if seed[-1] in ['a', 'b']:
        seed = seed[:-1]

    return int(seed)

def chalk_predictions(df, seed=42):
    """
    Baseline prediciton algorithm. Predict higher seeds will win every time
    Output = DF with Pred and Prob columns, indexed by input DF.
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
    Helper to score dataframe
    Output = DF with Pred and Prob columns, indexed by input DF.
    """
    # Produce Scores
    y_pred = model['clf'].predict(df[features])
    y_pred = pd.DataFrame(y_pred, columns=['Pred'], index=df.index)

    y_pred['Prob'] = y_pred['Pred'].apply(lambda x: norm(model['mean'], model['std']).cdf(x))

    y_pred['Pred'] = np.where(y_pred['Pred'] > 0, 1, 0)

    return y_pred

def score_round(matchups_r1, x_features, columns_key, scorer='chalk'):
    """
    Score matchups for round n and create matchups for round n+1
    """
    # Merge Current Round Matchups with X_Features
    score_r1 = createModelData(base=matchups_r1, x_features=x_features, columns_key=columns_key)

    # Score round matchups
    if scorer=='chalk':
        predictions = chalk_predictions(df=score_r1)
    else:
        predictions = model_predictions(df=score_r1, model=scorer, features=columns_key['features']) # ToDo - pass features?

    # Merge Predictions/Probabilites back to Score DF
    pred_r1 = score_r1.merge(predictions, left_index=True, right_index=True)

    # Identify Seed/TeamID of predicted Winner
    pred_r1['WTeamID_pred'] = np.where(pred_r1['Pred']==1, pred_r1['HTeamID'], pred_r1['ATeamID'])
    pred_r1['WSeed_pred'] = np.where(pred_r1['Pred']==1, pred_r1['Seed_H'], pred_r1['Seed_A'])

    # Get Slots
    slots_simple = pd.read_csv('{}/MNCAATourneySeedRoundSlots.csv'.format(base_data_path))
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

def fnScore(base, x_features, scorer='chalk'):
    """
    docstring
    """

    # ToDo - Split for simple scoring method?

    ## READ FEATURE DICT
    columns_key = getFeatureDict(pd.read_csv('{}/mmml/mmml/feature_list2.csv'.format(os.path.dirname(os.getcwd()))))

    #### Get Initial Set of True Results
    true_outcome = base.copy()
    true_outcome['WTeamID_true'] = np.where(true_outcome['HWin']==1, true_outcome['HTeamID'], true_outcome['ATeamID'])
    true_outcome = true_outcome[['Season', 'GameRound', 'GameSlot', 'WTeamID_true']]

    #### Create dictionary of base matchups and predictions for each round
    round_dict = {i:{'base':None,'pred':None} for i in range(1,7)}
    round_dict[1]={'base':base.query('GameRound==1')}

    #### Score each round and create next round matchups
    for round_num in range(1,7):
        logging.info("Getting predictions for Round {}...".format(round_num))
        base_r, pred_r0 = score_round(round_dict[round_num]['base'], x_features, columns_key, scorer=scorer)

        round_dict[round_num]['pred'] = pred_r0

        if round_num != 6:
            round_dict[round_num+1]['base'] = base_r

    #### CREATE OVERALL RESULTS DF
    logging.info("Creating DataFrame of all results...")
    results_df = round_dict[1]['pred']

    for i in range(2,7):
        results_df = results_df.append(round_dict[i]['pred'])

    results_df = results_df.merge(true_outcome, left_on=['Season', 'GameRound', 'GameSlot'], right_on=['Season', 'GameRound', 'GameSlot'], how='left')

    #### SCORE VS TRUE RESULTS
    logging.info("Scoring...")

    # Flag if prediction correct
    results_df['Correct'] = np.where(results_df['WTeamID_true'] == results_df['WTeamID_pred'], 1, 0)

    # Convert results to ESPN type score
    points_dict = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}
    results_df['Points'] = np.where(results_df['Correct']==1, results_df['GameRound'].apply(lambda x: points_dict[x]), 0)

    return round_dict, results_df

def fnEvaluate(results_df):
    results_df['Correct'] = np.where(results_df['WTeamID_true'] == results_df['WTeamID_pred'], 1, 0)

    points_dict = {1:10, 2:20, 3:40, 4:80, 5:160, 6:320}
    results_df['Points'] = np.where(results_df['Correct']==1, results_df['GameRound'].apply(lambda x: points_dict[x]), 0)

    year_list = list(set(results_df['Season']))
    year_list.sort()

    for year in year_list:
        acc = results_df.query('Season=={}'.format(year))['Correct'].sum() / results_df.query('Season=={}'.format(year))['Correct'].count()
        pts = results_df.query('Season=={}'.format(year))['Points'].sum()
        print("{year}: {acc}, {pts}".format(year=year, acc=acc, pts=pts))

def fnGetBracket(results_df, save):
    teams = pd.read_csv('{}/MTeams.csv'.format(base_data_path))

    merged = results_df.merge(teams, left_on='HTeamID', right_on='TeamID')\
                        .merge(teams, left_on='ATeamID', right_on='TeamID', suffixes=['_H', '_A'])

    bracket = merged[['Season', 'GameRound', 'Seed_H', 'Seed_A', 'TeamName_H', 'TeamName_A', 'Pred', 'Prob']].copy()

    bracket.sort_values(by=['Season', 'GameRound', 'Seed_H'], inplace=True)

    bracket.to_csv('{}/Output/{}.csv'.format(os.path.dirname(os.getcwd()), save))

    return bracket
