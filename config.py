# config.py

# File paths
DATA_PATH = "data/"
BALL_BY_BALL_FILE = DATA_PATH + "IPL_Ball_by_Ball_2008_2022.csv"
MATCHES_FILE = DATA_PATH + "IPL_Matches_2008_2022.csv"

# List of features to be used in the model
FEATURES = [
    'BattingTeam',
    'BowlingTeam',
    'City',
    'runs_left',
    'balls_left',
    'wickets_left',
    'target_score',
    'crr',
    'rrr'
]

# The target variable
TARGET = 'result'