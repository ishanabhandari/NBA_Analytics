import os
import pandas as pd

NBA_DATA = 'nba_data'
STANDINGS = os.path.join(NBA_DATA, 'standings')
SCORES = os.path.join(NBA_DATA, 'scores')
RESULTS = os.path.join(NBA_DATA, 'results')

df = pd.read_csv(os.path.join(RESULTS, "nba_games.csv"), index_col=0)
