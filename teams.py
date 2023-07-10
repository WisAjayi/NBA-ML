from nba_api.stats.static import teams

team = {
    'id': 'team_id',
    'full_name': 'full_name',
    'abbreviation': 'abbreviation',
    'nickname': 'nickname',
    'city': 'city',
    'state': 'state',
    'year_founded': 'year_founded',
}

TEAM = teams.find_team_by_abbreviation('LAL')
print(TEAM)


# help()
# _find_teams(regex_pattern, row_id)
# _get_team_dict(team_row)
# find_teams_by_full_name(regex_pattern)
# find_teams_by_state(regex_pattern)
# find_teams_by_city(regex_pattern)
# find_teams_by_nickname(regex_pattern)
# find_teams_by_year_founded(year)
# find_team_by_abbreviation(abbreviation)
# find_team_name_by_id(player_id)
# get_teams(regex_pattern, row_id)