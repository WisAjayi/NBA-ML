from flask import Flask, render_template,request,redirect, url_for
from ML import Multivariable

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def index():

    try:

        if request.method == 'POST':


            if 'first_name' and 'last_name' and 'team' in request.form:

                first = str(request.form.get('first_name')).capitalize().strip()
                last = str(request.form.get('last_name')).capitalize().strip()
                team = str(request.form.get('team')).capitalize().strip()

                Multivariable.PLAYER_FIRST_NAME = first
                Multivariable.PLAYER_LAST_NAME = last
                Multivariable.TEAM_NAME = team

                Name = Multivariable.init()
                data = Multivariable.generate_data()


                return render_template('index.html',
                                       name=Name,
                                       points_average=data['PTS'].mean(),
                                       points_assist=data['AST'].mean(),
                                       points_steal=data['STL'].mean(),
                                       points_block=data['BLK'].mean(),
                                       points_rebound=data['REB'].mean(),
                                       points_minutes=data['MIN'].mean(),
                                       points_turnover=data['TOV'].mean(),
                                       fgm=data['FGM'].mean(),
                                       fga=data['FGA'].mean(),
                                       fg3m=data['FG3M'].mean(),
                                       ftm=data['FTM'].mean(),
                                       fta=data['FTA'].mean(),
                                       points_most=data['PTS'].max(),
                                       assist_most=data['AST'].max(),
                                       steal_most=data['STL'].max(),
                                       block_most=data['BLK'].max(),
                                       rebound_most=data['REB'].max(),
                                       minutes_most=data['MIN'].max(),
                                       turnover_most=data['TOV'].max(),
                                       points_least=data['PTS'].min(),
                                       assist_least=data['AST'].min(),
                                       steal_least=data['STL'].min(),
                                       block_least=data['BLK'].min(),
                                       rebound_least=data['REB'].min(),
                                       minutes_least=data['MIN'].min(),
                                       turnover_least=data['TOV'].min(),)

    except KeyError as e:

        return render_template("index.html",error=e)

    except FileNotFoundError as e:

        return render_template("index.html",error=e)

    except Exception as e:

        return render_template("index.html",error=e)




    try:
            if request.method == 'POST':
                # Handle the first form submission
                return redirect(url_for('correlate'))

    except Exception as e:

        return render_template("index,html")




    return render_template('index.html',)



@app.route('/correlation',methods=['GET', 'POST'])
def correlate():

    Compare = "Stats being Compared:"
    Correlation = "Correlation:"

    try:

        if request.method == 'POST':

            if 'first_' and 'last_' and 'Stat_1' and 'teamm' and 'Stat_2' in request.form:


                first = str(request.form.get('first_')).capitalize().strip()
                last = str(request.form.get('last_')).capitalize().strip()
                team = str(request.form.get('teamm')).capitalize().strip()

                Stat_1 = str(request.form.get('Stat_1'))
                Stat_2 = str(request.form.get('Stat_2'))


                Multivariable.PLAYER_FIRST_NAME = first
                Multivariable.PLAYER_LAST_NAME = last
                Multivariable.TEAM_NAME = team

                Multivariable.generate_data()

                Multivariable.Name_Abbrev = Stat_1
                Multivariable.Compare_Abbrev = Stat_2
                print(Multivariable.corr())

                return render_template('correlate.html',
                                stat1=Stat_1,
                                stat2=Stat_2,
                                corr=Multivariable.corr(),
                                compare=Compare,
                                correlate=Correlation
                                )

    except KeyError as e:

        return render_template("correlate.html",error=e)

    except FileNotFoundError as e:

        return render_template("correlate.html",error=e)

    except Exception as e:


        return render_template("correlate.html",error=e)


    return render_template('correlate.html')

if __name__ == '__main__':
    app.run()
