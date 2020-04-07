import pandas as pd
import numpy as np
import math
import sys

def win_probs(*, home_elo, road_elo, hca_elo):
    """
    Home and road team win probabilities implied by Elo ratings and home court adjustment.
    """
    h = math.pow(10, home_elo/400)
    r = math.pow(10, road_elo/400)
    a = math.pow(10, hca_elo/400)
    denom = r + a*h
    home_prob = a*h / denom
    road_prob = r / denom
    return home_prob, road_prob

def update(*, winner, home_elo, road_elo, hca_elo, k, probs=False):
    """
    Update Elo ratings for a given match up.
    """
    home_prob, road_prob = win_probs(home_elo=home_elo, road_elo=road_elo, hca_elo=hca_elo)
    if winner[0].upper() == 'H':
        home_win = 1
        road_win = 0
    elif winner[0].upper() in ['R', 'A', 'V']: # road, away or visitor are treated as synonyms
        home_win = 0
        road_win = 1
    else:
        raise ValueError('unrecognized winner string', winner)
    new_home_elo = home_elo + k*(home_win - home_prob)
    new_road_elo = road_elo + k*(road_win - road_prob)
    if probs:
        return new_home_elo, new_road_elo, home_prob, road_prob
    else:
        return new_home_elo, new_road_elo

def getNumericSeed(seed):
    seed = seed[1:].lstrip("0")

    if seed[-1] in ['a', 'b']:
        seed = seed[:-1]

    return int(seed)

def update_progress_bar(current, total):
    """
    Prints progress inplace. To be used with iterative function.
    :param current: numeric. Current stage
    :param total: numeric. Total number of stages (end goal)
    :return: None
    """
    barLength = 24  # Modify this to change the length of the progress bar
    progress = float(current) / float(total)
    block = int(round(barLength * progress))

    text = "\rProgress: [{block}]: {current} / {total}".format(
        block=("=" * max((block - 1), 0) + ">" + " " * (barLength - block)), current=current, total=total)
    sys.stdout.write(text)
    sys.stdout.flush()

class gameResults:
    def __init__(self, df):
        self.df = df

    def toHomeAwayFormat(self, seed=42):
        """
        Takes DF of game results with WTeam/LTeam format and returns DF of game results scrambled with HomeTeam/AwayTeam format.
        Adds column of AWin (binary indicator)
        """
        df = self.df
        np.random.seed(seed=seed)
        df['rand'] = np.random.rand(df.shape[0])
        df['NLoc'] = np.where(df['WLoc'] == "N", 1, 0)

        df_H = df.query('WLoc == "H" or (WLoc == "N" and rand > 0.5)')
        df_A = df.query('WLoc == "A" or (WLoc == "N" and rand <= 0.5)')

        df_H = df_H.drop('WLoc', axis=1)
        for col in list(df_H.columns).copy():
            if col[0] == "W":
                df_H = df_H.rename(columns={col:('H'+col[1:])})

            if col[0] == "L":
                df_H = df_H.rename(columns={col:('A'+col[1:])})
        df_H['HWin'] = 1
        df_H['AWin'] = 0

        # Flip W/L to B/A for half of games
        df_A = df_A.drop('WLoc', axis=1)
        for col in list(df_A.columns).copy():
            if col[0] == "W":
                df_A = df_A.rename(columns={col:('A'+col[1:])})

            if col[0] == "L":
                df_A = df_A.rename(columns={col:('H'+col[1:])})

        df_A['HWin'] = 0
        df_A['AWin'] = 1

        home_away_results = df_H.append(df_A, sort=True)

        return home_away_results

    def toSeasonAggFormat(self):
        """
        Duplicate results from raw results DF to allow for season-aggregations. Transforms H/A view to Team/Opponent.
        Each game is represented 2x in resulting DF (once in terms of Team A and once in terms of Team B)
        """
        df_a = self.toHomeAwayFormat().copy()
        df_b = df_a.copy()

        for col in list(df_a.columns).copy():
            if col[0] == "H":
                df_a = df_a.rename(columns={col:(col[1:])})

            if col[0] == "A":
                df_a = df_a.rename(columns={col:('Opp'+col[1:])})

            #df_a['Win'] = 1

        for col in list(df_b.columns).copy():
            if col[0] == "H":
                df_b = df_b.rename(columns={col:('Opp'+col[1:])})

            if col[0] == "A":
                df_b = df_b.rename(columns={col:(col[1:])})

        df_c = df_a.append(df_b, sort=True)

        return df_c

    def getSeasonStats(self):
        """
        Aggregate game by game stats to the Team/Season level. Calculate various advanced stats
        """
        df_season_agg = self.toSeasonAggFormat()

        # Calculate Possessions for each game
        df_season_agg['possessions'] = 0.5 * (df_season_agg['FGA']  + 0.475 * df_season_agg['FTA'] - df_season_agg['OR'] + df_season_agg['TO']) \
                        + 0.5 * (df_season_agg['OppFGA'] + 0.475 * df_season_agg['OppFTA'] - df_season_agg['OppOR'] + df_season_agg['OppTO'])

        # Aggregate to Season Summary Level
        season_stats = df_season_agg.groupby(['TeamID', 'Season']).sum()

        season_stats = season_stats.rename(columns={'Win':'wins'})

        # Season Advanced Stats
        season_stats['o_eff'] = season_stats['Score'] / season_stats['possessions'] * 100
        season_stats['d_eff'] = season_stats['OppScore'] / season_stats['possessions'] * 100
        season_stats['net_eff'] = season_stats['o_eff'] - season_stats['d_eff']

        season_stats.drop('DayNum', axis=1, inplace=True)
        season_stats.drop('OppTeamID', axis=1, inplace=True)
        season_stats.drop('rand', axis=1, inplace=True)

        return season_stats

    def getElo(self, hca_elo=65, k=20, initial_elo=1500.0):
        # Transform W/L to H/A results format
        home_away_results = self.toHomeAwayFormat()

        # Define Initialized Elo Dict
        elo_dict = {}
        for season in set(home_away_results['Season']):
            HTeams = list(set(home_away_results.query('Season == {}'.format(season))['HTeamID']))
            ATeams = list(set(home_away_results.query('Season == {}'.format(season))['ATeamID']))

            teams_list = list(set(HTeams + ATeams))
            elo_dict[season] = {j:[initial_elo] for j in teams_list}

        # Ensure game results are sorted
        home_away_results.sort_values(['Season', 'DayNum'], inplace=True)

        # Iterate through game results, updating elo dict
        current_progress = 0
        for i in home_away_results.index:
            update_progress_bar(current_progress, home_away_results.shape[0])

            # Get TeamID's and Season
            HTeamID = int(home_away_results.loc[i]['HTeamID'])
            ATeamID = int(home_away_results.loc[i]['ATeamID'])
            season = int(home_away_results.loc[i]['Season'])

            # Determine true winner
            if home_away_results.loc[i]['HWin']==1:
                winner = "H"
            else:
                winner = "A"

            # Previous Elo Scores
            home_elo_initial = elo_dict[season][HTeamID][-1]
            away_elo_initial = elo_dict[season][ATeamID][-1]

            # Calculate Elo update, accounting for Home Court Advantage
            if home_away_results.loc[i]['NLoc'] == 1:
                # No home court advantage at neutral site
                h_elo_update, a_elo_update = update(winner=winner, home_elo=home_elo_initial, road_elo=away_elo_initial, hca_elo=0, k=k)
            else:
                h_elo_update, a_elo_update = update(winner=winner, home_elo=home_elo_initial, road_elo=away_elo_initial, hca_elo=hca_elo, k=k)

            # Update Elo scores in dict
            elo_dict[season][HTeamID].append(h_elo_update)
            elo_dict[season][ATeamID].append(a_elo_update)

            current_progress = current_progress + 1

        # Convert Nested elo Dict to DataFrame
        elo_df = pd.DataFrame(pd.DataFrame.from_dict(elo_dict).stack(), columns=['elo'])
        elo_df['last_elo'] = elo_df['elo'].apply(lambda x: x[-1])
        elo_df.index.set_names(['TeamID', 'Season'], inplace=True)

        return elo_df

    def getBase(self):
        """
        Format DataFrame for use as base of model data w/ key info only.
        To be used on Tournament results
        """
        home_away_results = self.toHomeAwayFormat()
        base = home_away_results[['HTeamID', 'ATeamID', 'Season','DayNum','HWin', 'HScore', 'AScore']]
        return base

#     def getTourneySeedWinPct(self, seeds, current_season):
#         tourney_results = self.df.query('Season < {}'.format(current_season))
#         seeds['numeric_seed'] = seeds['Seed'].apply(lambda x: getNumericSeed(x))

#         results_seeded = tourney_results.merge(seeds, left_on=['WTeamID', 'Season'], right_on=['TeamID', 'Season'], how='left')\
#         .merge(seeds, left_on=['LTeamID', 'Season'], right_on=['TeamID', 'Season'], how='left', suffixes=('_W', '_L'))[['Season', 'numeric_seed_W', 'numeric_seed_L']]

#         wins = results_seeded.pivot_table(index='numeric_seed_W', columns='numeric_seed_L', aggfunc=np.count_nonzero)

#         results_seeded_reverse = results_seeded.copy().rename(columns={'numeric_seed_W':'numeric_seed_L1','numeric_seed_L':'numeric_seed_W1'})
#         results_seeded_reverse = results_seeded_reverse.rename(columns={'numeric_seed_W1':'numeric_seed_W','numeric_seed_L1':'numeric_seed_L'})
#         stacked = results_seeded[['Season', 'numeric_seed_W', 'numeric_seed_L']].append(results_seeded_reverse[['Season', 'numeric_seed_W', 'numeric_seed_L']], sort=True)

#         games = stacked.pivot_table(index='numeric_seed_W', columns='numeric_seed_L', aggfunc=np.count_nonzero)

#         win_pct = wins/games
#         np.fill_diagonal(win_pct.values, -999)
#         win_pct.fillna(-999, inplace=True)

#         games.fillna(-999, inplace=True)

#         return win_pct, games
