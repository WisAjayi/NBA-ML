import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import TEAM_NAME,PLAYER,Name,Name_Abbrev,Compare,Compare_Abbrev


METRICS = ["Correlation", "Regression", "P-Values & Coefficients", "Bayesian Information Criterion"]
STATS = ["PTS = Points", "AST = Assist", "STL = Steal", "BLK = Block", "REB = Rebound","TOV = Turnover",] # Stats Can be used / Changed for Correlation #

def clean_for_correlation():
    
    new_df = data.drop(['SEASON_ID', 'Player_ID','Game_ID', 'GAME_DATE', 'MATCHUP','WL', 'VIDEO_AVAILABLE'], axis=1) # Unnecessary Columns #
    return new_df


def clean_max_point():
    
    max_pts_index = data['PTS'].idxmax()
    print(new_data.loc[[max_pts_index]].to_string(index=False), "These are stats of the game.")


def clean_min_point():
    
    min_pts_index = data['PTS'].idxmin()
    print(new_data.loc[[min_pts_index]].to_string(index=False),"These are stats of the game.")





data = pd.read_csv(f'/Users/wisdomajayi/Documents/Working Directory/NBA/TEAMS/{TEAM_NAME}/GAMELOG/{PLAYER}.csv')
new_data = clean_for_correlation() ### Get New Data without unnecessary columns ###


print("Average Score = ",data['PTS'].mean())

print("Least Points Scored  = ", data['PTS'].min())
clean_min_point()

print("Most Points Scored = ", data['PTS'].max())
clean_max_point()

print(f"The Correlation of {Name} to {Compare} Played = ", data[f'{Name_Abbrev}'].corr(data[f'{Compare_Abbrev}']) )

  


# CREATE MASK MATRIX #   
mask = np.zeros_like(new_data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
#print("The mask Data = ",mask)
    
    

  
### TRAINING & TEST DATA SPLIT ###
points = new_data['PTS']
otherstats = new_data.drop('PTS', axis=1)
X_train, X_test, y_train, y_test = train_test_split(otherstats, points, test_size=0.2, random_state=50)
print(len(X_train)/len(otherstats))
print( X_test.shape[0]/otherstats.shape[0] )


def visualise_pts():
    
    plt.figure(figsize=(10, 6))
    plt.hist(data['PTS'], bins=len(data['PTS']), ec='red', color='#2196f3')
    plt.ylabel('PTS in Game')
    plt.xlabel('Number of Games played')
    plt.ylim(0,100 )
    #plt.xlim(0,100)
    plt.show()
    
    
    
    # Second Graph #
    plt.figure(figsize=(10, 6))
    sns.distplot(data['PTS'], bins=len(data['PTS']), hist=True, kde=False, color='#fbc02d') ### Deprecation Error Given, Remember to change function displot() in the future ###
    plt.show()


    # Third Graph #
    plt.figure(figsize=(10, 6))
    plt.hist(data['PTS'], ec='black', color='#00796b')
    plt.show()




def visualise_correlation():
    
    plt.figure(figsize=(20,6))
    sns.heatmap(new_data.corr(), mask=mask, annot=True, annot_kws={"size": 14})
    sns.set_style('white')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()



### MULTIVARIABLE REGRESSION ###

def _regression():

    regr = LinearRegression()
    regr.fit(X_train, y_train)

    # print out r-squared for training and test datasets
    print('Training data r-squared:', regr.score(X_train, y_train))
    print('Test data r-squared:', regr.score(X_test, y_test))
    print('Intercept', regr.intercept_)
    print( pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef']))




# P Values & Evaluating Coefficients #
def p_value_and_coefficient():
   
    X_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_const)
    results = model.fit()
    print ( pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)}) )



def multicol():
    
    X_const = sm.add_constant(X_train)
    
    #Testing for Multicollinearity #
    print( variance_inflation_factor(exog=X_const.values, exog_idx=1) )

    # A for loop that prints out all the VIFs for all the features
    for i in range(X_const.shape[1]):
        print(variance_inflation_factor(exog=X_const.values, exog_idx=i))
    print('All done!')


    vif = [variance_inflation_factor(exog=X_const.values, exog_idx=i) for i in range(X_const.shape[1])]
    print(pd.DataFrame({'coef_name': X_const.columns, 'vif': np.around(vif, 2)}))





def bic():
    
    # MODEL SIMPLIFICATION & THE BIC #
    # Bayesian Information Criterion #
    
    
    # Original model with log prices and all features
    X_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_const)
    results = model.fit()
    org_coef = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})
    print('BIC is', results.bic)
    print('r-squared is', results.rsquared)




    # Reduced model #1 excluding Turnovers
    X_const = sm.add_constant(X_train)
    X_const = X_const.drop(['TOV'], axis=1)
    model = sm.OLS(y_train, X_const)
    results = model.fit()
    coef_minus_tov = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})
    print('BIC is', results.bic)
    print('r-squared is', results.rsquared)




    # Reduced model #2 excluding Turnovers & Minutes Played. 
    X_const = sm.add_constant(X_train)
    X_const = X_const.drop(['TOV', 'MIN'], axis=1)
    model = sm.OLS(y_train, X_const)
    results = model.fit()
    reduced_coef = pd.DataFrame({'coef': results.params, 'p-value': round(results.pvalues, 3)})
    print('BIC is', results.bic)
    print('r-squared is', results.rsquared)




    frames = [org_coef, coef_minus_tov, reduced_coef]
    print( pd.concat(frames, axis=1) )




# Data Transformation #
# Data Transform with _skew() and log_skew().
def _skew():

    print( data['PTS'].skew() )




def log_skew():
    
    y_log = np.log(data['PTS'])
    print ( y_log.tail() )
    print( y_log.skew() )