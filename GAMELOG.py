from nba_api.stats.endpoints.playergamelog import PlayerGameLog
from nba_api.stats.library.parameters import SeasonAll
import pandas as pd


CURRENT_TEAM = ""
PLAYER_NAME = ""
PLAYER_ID = ""

current_player = PlayerGameLog(player_id=PLAYER_ID,season=SeasonAll.all).get_data_frames()
current_player[0].to_csv(f'TEAMS/{CURRENT_TEAM}/GAMELOG/{PLAYER_NAME}.csv', index=False)


### Column Names Of Dataframe ###
def all_columns():
    
    column_names = current_player[0].columns.tolist()
    for name in column_names:
        print(name)