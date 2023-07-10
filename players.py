from nba_api.stats.static import players
import json
import time


player = {
    'id': 'player_id',
    'full_name': 'full_name',
    'first_name': 'first_name',
    'last_name': 'last_name',
    'is_active': True or False
}



def player_list():
    
    ALL_PLAYERS = players.get_active_players()

    for i in ALL_PLAYERS:
        
        print(i['id'])
        print(i['full_name'])
        time.sleep(3)
  

P = players.find_players_by_full_name("")
print(P)

# help()   
# _find_players(regex_pattern, row_id)
# _get_player_dict(player_row)
# find_players_by_full_name(regex_pattern)
# find_players_by_first_name(regex_pattern)
# find_players_by_last_name(regex_pattern)
# find_player_by_id(player_id)
# get_players()
# get_active_players()
# get_inactive_players()