from nba_api.stats.endpoints import playercareerstats
import time


# Giannis 
career = playercareerstats.PlayerCareerStats(player_id='203507') 
print(career)

# pandas data frames
df = career.get_data_frames()[0]
print(df)



column_names = df.columns.tolist()
for name in column_names:
    print(name)


# json
#print(career.get_json())

# dictionary
#print(career.get_dict())