from nba_api.stats.endpoints import leagueleaders
import pandas as pd
import inspect

# Pull data for the top 500 scorers by PTS column
top_500 = leagueleaders.LeagueLeaders(
    per_mode48='PerGame',
    season='2020-21',
    season_type_all_star='Regular Season',
    stat_category_abbreviation='PTS'
).get_data_frames()[0][:500]

#print(inspect.signature(leagueleaders.LeagueLeaders)) # Inspect is better.
#print(help(leagueleaders.LeagueLeaders))
print(top_500)




#top_500["PLAYER_ID"] = pd.to_numeric(top_500["PLAYER_ID"], downcast="float")
#print(help(top_500.groupby))
#top_500['PLAYER_ID'] = top_500['PLAYER_ID'].astype(int)
print(type(top_500['PLAYER_ID']))
