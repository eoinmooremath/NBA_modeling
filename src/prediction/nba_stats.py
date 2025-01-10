import requests
from time import sleep
import pandas as pd
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com',
    'Upgrade-Insecure-Requests': '1'
}

def make_info():
    # this function scrapes NBA.com for all the basic biographical information (height, weight, team) for every player currently in the NBA, and saves it in a pandas df
    url_bio ='https://stats.nba.com/stats/leaguedashplayerbiostats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&Season=2024-25&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='
    url_position = 'https://stats.nba.com/stats/playerindex?College=&Country=&DraftPick=&DraftRound=&DraftYear=&Height=&Historical=1&LeagueID=00&Season=2024-25&SeasonType=Regular%20Season&TeamID=0&Weight='
    response_bio = requests.get(url_bio, headers=headers)
    sleep(1.6)
    response_pos = requests.get(url_position, headers=headers)
    if response_pos.status_code == 200 and response_bio.status_code==200:
        data_bio = response_bio.json()
        columns_bio = data_bio['resultSets'][0]['headers']
        rows_bio = data_bio['resultSets'][0]['rowSet']
        df_bio = pd.DataFrame(rows_bio, columns=columns_bio)
        
        data_pos = response_pos.json()  
        columns_pos = data_pos['resultSets'][0]['headers']
        rows_pos = data_pos['resultSets'][0]['rowSet']
        df_pos = pd.DataFrame(rows_pos, columns = columns_pos)
        df_pos=df_pos[ ['PERSON_ID', 'POSITION']]
        
        df_info = pd.merge(df_bio, df_pos, how = 'left',left_on = 'PLAYER_ID', right_on = 'PERSON_ID')
        df_info['DRAFT_YEAR']=df_info['DRAFT_YEAR'].apply(pd.to_numeric,errors='coerce').fillna(2024)
        df_info['DRAFT_ROUND'] =df_info['DRAFT_ROUND'].apply(pd.to_numeric,errors='coerce').fillna(3)
        df_info['years_in_league'] = 2024-df_info['DRAFT_YEAR']

        df_info = df_info[[ 'PLAYER_NAME','PLAYER_ID','TEAM_ABBREVIATION', 'PLAYER_HEIGHT_INCHES','PLAYER_WEIGHT',
                        'years_in_league', 'AGE', 'DRAFT_ROUND','POSITION' ]  ]
        df_info['POSITION']=df_info['POSITION'].replace('G','point guard shooting guard')
        df_info['POSITION']=df_info['POSITION'].replace('F' , 'small forward power forward')
        df_info['POSITION']=df_info['POSITION'].replace('C' , 'center')
        df_info['POSITION']=df_info['POSITION'].replace('G-F' , 'point guard shooting guard small forward power forward')
        df_info['POSITION']=df_info['POSITION'].replace('F-G' , 'point guard shooting guard small forward power forward')
        df_info['POSITION']=df_info['POSITION'].replace('F-C' , 'small forward power forward center')
        df_info['POSITION']=df_info['POSITION'].replace('C-F' , 'small forward power forward center' )
        df_info = df_info.reset_index(drop=True)    
        return df_info

def get_players(n_games=6, how='general'):
    df_info = make_info()
    #the string below will get the stats over the last n games for all current nba players
    url_stats = f'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames={n_games}&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2024-25&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='
    response_stats = requests.get(url_stats, headers=headers)
    data_stats = response_stats.json()
    columns_stats = data_stats['resultSets'][0]['headers']
    rows_stats = data_stats['resultSets'][0]['rowSet']
    df_stats = pd.DataFrame(rows_stats, columns=columns_stats)
    df_stats_detailed = df_stats.loc[:,['PLAYER_NAME','PLAYER_ID','MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','AST','STL','BLK','TOV','PF','PTS','PLUS_MINUS']]
    df_stats_detailed = df_stats_detailed.rename(columns={'MIN':'minutes','FGM':'fieldGoalsMade','FGA':'fieldGoalsAttempted',
                                        'FG3M':'threePointersMade',
                                        'FG3A':'threePointersAttempted',
                                        'FTM':'freeThrowsMade','FTA':'freeThrowsAttempted',
                                        'OREB':'reboundsOffensive','DREB':'reboundsDefensive',
                                        'AST':'assists','STL':'steals','BLK':'blocks','TOV':'turnovers',
                                        'PF':'foulsPersonal','PTS':'points','PLUS_MINUS':'plusMinusPoints' })
    df_stats_general = df_stats_detailed.loc[:, ['PLAYER_NAME', 'PLAYER_ID','points', 'threePointersMade', 'steals', 'blocks', 'turnovers', 'assists','minutes']]
    df_stats_general['rebounds']=df_stats_detailed['reboundsOffensive']+df_stats_detailed['reboundsDefensive']

    df_info=df_info.rename(columns={'PLAYER_WEIGHT':'weight', 'PLAYER_HEIGHT_INCHES':'height', 'AGE':'age', 'DRAFT_ROUND':'draft_round', 'POSITION':'position'})
    df_info['home?']=0
    df_info = df_info[['PLAYER_NAME', 'PLAYER_ID', 'weight','height','age', 'years_in_league', 'home?','draft_round', 'position', 'TEAM_ABBREVIATION']]
    df_info['draft_round'] = df_info['draft_round'].replace(0, 3)
    df_all_general = pd.merge(df_stats_general, df_info, how = 'inner', on = ['PLAYER_NAME', 'PLAYER_ID'])
    df_all_general=df_all_general.rename(columns={'PLAYER_NAME':'name', 'PLAYER_ID':'player_id', 'TEAM_ABBREVIATION':'team_id'})
    df_all_detailed = pd.merge(df_stats_detailed,df_info, how='inner', on=['PLAYER_NAME', 'PLAYER_ID'])
    df_all_detailed=df_all_detailed.rename(columns={'PLAYER_NAME':'name', 'PLAYER_ID':'player_id', 'TEAM_ABBREVIATION':'team_id'})
    if how =='general':
        return df_all_general
    elif how == 'detailed':
        return df_all_detailed
    else:
        return (df_all_general, df_all_detailed)  
    
def matchup(game_id, df_all, out_players, m_players=6):  
    df_all =  df_all[ ~df_all['name'].isin(out_players)]
    team_id_home = game_id.split(' @ ')[0]
    team_id_away = game_id.split(' @ ')[1]
    team_home = df_all.loc[df_all['team_id'] == team_id_home]
    team_home.loc[:,'home?']=1
    team_away = df_all.loc[df_all['team_id'] == team_id_away]
    team_home = team_home.sort_values(by='minutes', ascending=False).iloc[:m_players,:]
    team_away = team_away.sort_values(by='minutes', ascending=False).iloc[:m_players,:]
    both_teams = pd.concat([team_home,team_away],axis=0)
    both_teams = both_teams.sample(frac=1)
    both_teams = both_teams.reset_index(drop=True)
    return both_teams 