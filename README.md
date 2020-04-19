# MarchMadnessML
This is a project to create a simple machine learning model to generate bracket predictions for the NCAA basketball tournament.
This project was originated for the 2018 tournament and revamped for 2020.

Data can be sourced from the Kaggle competition: https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament

## Methodology
- <b>Dependent Variable:</b> Score differential (Team 1 Points - Team 2 Points)
- <b>Independent Attributes:</b> Offensive & defensive efficiency metrics, ELO,
team rankings such as KenPom
- <b>Model:</b> XGBoost Regression (5 fold CV)
- <b>Development / OOT Seasons:</b> [2003 - 2016] / [2017 - 2019]

## Performance
| Season | Accuracy (Baseline) | Points (Baseline) | Percentile (Baseline) |
|--------|---------------------|-------------------|-----------------------|
| 2017   | 63.4% (71.4%)       | 910 (980)         | -                     |
| 2018   | 63.3% (55.5%)       | 1120 (650)        | -                     |
| 2019   | 68.3% (65.0%)       | 950 (920)         | 88.1 (86.3)           |

Baseline determined by picking higher seed each game, randomly choosing a winner
in the case of equal seeds. Median results from 1000 iterations displayed.

## Getting Started
```
python -m venv MarchMadnessML
source MarchMadnessML/bin/activate
pip install -r requirements.txt
python driver.py
```

## Project Structure
```
MarchMadnessML
├── 2018_Archive
├── Data
│   ├── Processed
│   └── Raw
│       └── google-cloud-ncaa-march-madness-2020-division-1-mens-tournament
│           ├── MDataFiles_Stage1
│           │   ├── Cities.csv
│           │   ├── Conferences.csv
│           │   ├── MConferenceTourneyGames.csv
│           │   ├── MGameCities.csv
│           │   ├── MMasseyOrdinals.csv
│           │   ├── MNCAATourneyCompactResults.csv
│           │   ├── MNCAATourneyDetailedResults.csv
│           │   ├── MNCAATourneySeedRoundSlots.csv
│           │   ├── MNCAATourneySeeds.csv
│           │   ├── MNCAATourneySlots.csv
│           │   ├── MRegularSeasonCompactResults.csv
│           │   ├── MRegularSeasonDetailedResults.csv
│           │   ├── MSeasons.csv
│           │   ├── MSecondaryTourneyCompactResults.csv
│           │   ├── MSecondaryTourneyTeams.csv
│           │   ├── MTeamCoaches.csv
│           │   ├── MTeamConferences.csv
│           │   ├── MTeamSpellings.csv
│           │   └── MTeams.csv
│           ├── MEvents2015.csv
│           ├── MEvents2016.csv
│           ├── MEvents2017.csv
│           ├── MEvents2018.csv
│           ├── MEvents2019.csv
│           ├── MPlayers.csv
│           └── MSampleSubmissionStage1_2020.csv
├── Model_Objects
├── Notebooks
│   ├── 01-DataPrep.ipynb
│   ├── 02-SplitData.ipynb
│   ├── 03-FeatureProcessing.ipynb
│   ├── 04_Train.ipynb
│   ├── 05_Predict.ipynb
│   └── 06_Baseline-Scores.ipynb
├── Output
├── README.md
├── Scripts
│   ├── data_prep.py
│   ├── driver.py
│   ├── feature_processing.py
│   ├── score.py
│   ├── split_data.py
│   └── train.py
├── mmml
│   └── mmml
│       ├── __init__.py
│       ├── config.py
│       ├── feature_list2.csv
│       ├── game_results.py
│       └── utils.py
└── requirements.txt
```
