import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from config import PLAYER, TEAM_NAME, TEAM_ABBREV as TEAM ### Use to search for Team Match upn to gather data ###


data = pandas.read_csv(f'../TEAMS/{TEAM_NAME}/GAMELOG/{PLAYER}.csv')
#print(data.describe())

X = DataFrame(data, columns=['PTS'])
y = DataFrame(data, columns=['FGM'])
matchups = DataFrame(data, columns=['MATCHUP'])

REGRESSION = None

def visualize():
    
    plt.figure(figsize=(10,6))
    plt.scatter(X, y, alpha=0.5)
    plt.title('PTS vs FGM')
    plt.xlabel('Points')
    plt.ylabel('Field Goal Made')
    plt.ylim(0, 30)
    plt.xlim(0, 50)
    plt.show()


def regression_stats():
    
    ### Compare Points to Field Goal Made ###
    regression = LinearRegression()
    regression.fit(X, y)
    print("Slope Coefficient = ", regression.coef_)   # Slope coefficient
    print("Intercept = ", regression.intercept_) # Intercept
    print("Prediction for X = ",regression.predict(X))
    print("R Square = ", regression.score(X, y) ) #Getting r square from Regression



def clean_matchup_index():
    
    ### Get VS Specific team after reading pandas series ###
    Indexes = []
    rows_with_team = matchups[matchups['MATCHUP'].str.contains(f'{TEAM}')]
    print(rows_with_team)
    
    # Print the indexes of the matching rows
    print(f"Indexes of rows containing '{TEAM}':")
    for index in rows_with_team.index:
        Indexes.append(index)
        print(index)

    # Print the total count of rows
    total_rows = len(rows_with_team)
    print("\nTotal count of rows:", total_rows)
    
    return Indexes
   


def index_pandaframe():
    
    ### Creates a Pandas dataframe with the indexes of the teams we previously searched ###
    new_df = data.loc[INDEXES].reset_index(drop=True)
    print(new_df)

INDEXES = clean_matchup_index()   
index_pandaframe()