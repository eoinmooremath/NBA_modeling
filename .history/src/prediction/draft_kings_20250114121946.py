    
import pickle
from datetime import datetime
import copy
from tqdm import tqdm 
import pandas as pd
import numpy as np
from dateutil import parser
import isodate
from datetime import timedelta 
from sklearn.model_selection import train_test_split
import re 
import pytz 
import requests
import json
from bs4 import BeautifulSoup
import numpy as np
import pickle
import os
import pandas as pd 
import numpy as np
from time import sleep
import unicodedata


def get_contests(target_date = False):
    url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
    response = requests.get(url)
    DK= response.json()
    contest_list = DK['Contests']
    
    def transform_contest(contest):
        #TODO add a column for the number of entrants (so I can enter bigger contests.) Also, Fifty Fifty is not working
        keep_cols = ['n', 'a', 'po', 'sd', 'id', 'gameType', 'dg','nt','m']  #'attr' if ;attr contains isdoubleup, IsFiftyfifty
        contest_edit= {k:v for k,v in contest.items() if k in keep_cols}
        if 'IsDoubleUp' in contest['attr'].keys():
            contest_edit['Double Up'] = ( contest['attr']['IsDoubleUp'].lower() == 'true')
        else: 
            contest_edit['Double Up'] = False
        if 'IsFiftyFifty' in contest['attr'].keys():
            contest_edit['Fifty Fifty'] = ( contest['attr']['IsFiftyfifty'].lower() == 'true')
        else:
            contest_edit['Fifty Fifty'] = False
        
        #we need the following to convert their start time into something human-readable
        date_string = contest_edit['sd']
        numbers = re.findall(r'\d+', date_string)
        extracted_number = int(numbers[0]) if numbers else None
        contest_edit['sd'] = datetime.fromtimestamp(extracted_number/1000)
        site = f"www.draftkings.com/draft/contest/{contest_edit['id']}"
        #link = f'<a href="{site}" target="_blank">{site}</a>'       
        #contest_edit['URL'] = f"www.draftkings.com/draft/contest/{contest_edit['id']}"
        contest_edit['URL'] = site
        return contest_edit

    contests = pd.DataFrame( [transform_contest(contest) for contest in contest_list] )
    contests = contests.rename(columns = {'n':'Contest Name', 'a':'Entry Fee', 'po': 'Prize', 'sd': 'Start Time', 'nt':'Current Entrants', 'm':'Max Entrants','id':'ID', 'gameType':'Game Type', 'dg':'Draft Groups'})
    contests['Remaining Entries'] = contests['Max Entrants'] - contests['Current Entrants']
    if target_date:
        pacific_tz = pytz.timezone('America/Los_Angeles')
        if target_date.lower() =='today':
            target_date = datetime.now(pacific_tz).date()
        elif target_date.lower() in ['tomorrow', 'tmrw']:
            target_date = (datetime.now(pacific_tz)+timedelta(days=1)).date() # Get tomorrow's date
        else:
            target_date = pd.to_datetime(target_date).date()
        matching_dates = contests['Start Time'].apply(lambda x: x.date() == target_date)  # Use .dt.date for comparison
        return contests.loc[matching_dates,:]
    return contests

def get_draft_groups(target_dates=False):
    contests = get_contests(target_dates)

    draft_group_ids = contests['Draft Groups'].unique()
    my_draft_groups = []
    for id in draft_group_ids:
        #return an array of dataframes. each dataframe is a draftgroup. 
        player_data=[]
        url = f'https://api.draftkings.com/draftgroups/v1/draftgroups/{id}/draftables'
        response = requests.get(url)
        game_data= response.json()  
        for player in game_data['draftables']:
            name = player['displayName']
            position = player['position']
            if 'salary' in player:
                salary = player['salary']
            else:
                salary=0
            status = player['status']
            team =  player['teamAbbreviation']
            game_name = player['competition']['name']
            game_teams = game_name.split('@')
            home = team in game_teams[1]
            if home:
                opponent = game_teams[0]
            else:
                opponent = game_teams[1]    
            game_Date = player['competition']['startTime'].split('-')
            game_Date = f'{game_Date[0]}-{game_Date[1]}-{game_Date[2][:2]}'
            competitions=''
            for comp in player['competitions']:
                competitions+= f",{comp['competitionId']}"
            competitions= competitions[1:] #delete the ',' in front of competitions
            # player_data.append((name, position, salary, team,  home, game_Date, opponent, status, competitions, id))
            player_dict = {'Name':name, 'Position':position, 'Salary':salary, 'Game': game_name, 'Team':team,'Home?':home, 'Date':game_Date, 'Opponent':opponent, 'Status':status,'Competitions':competitions, 'Draft Group':id}
            
            player_series = pd.Series(player_dict)
            player_data.append(player_series)
        draft_group = pd.DataFrame(player_data)
        print(draft_group)
        # draft_group =pd.Series(player_data, columns=['Name', 'Position', 'Salary', 'Team','Home?', 'Date', 'Opponent', 'Status','Competitions', 'Draft Group'])
        
        
        # For some draft_groups, players have two possible salaries. The contests we play in all use the lower salary. We keep this row only.
        draft_group = draft_group.loc[draft_group.groupby('Name')['Salary'].idxmin()]
        draft_group = draft_group.reset_index(drop=True)

        if not draft_group.empty and not draft_group['Salary'].eq(0).all():
            my_draft_groups.append(draft_group)
        if not draft_group.empty:
            if not draft_group['Salary'].eq(0).all():  # exclude drafts where players dont have a salary. These are different kinds of drafts we are not interested in.
                my_draft_groups.append(draft_group)
                
            
    return my_draft_groups


def clean_name(input_str):
    nfkd_form = unicodedata.normalize('NFD', input_str)
    no_accents =  ''.join([c for c in nfkd_form if not unicodedata.combining(c) and c.isalpha()])
    return no_accents.lower()


def get_rosters_wiki():
    
    # Function to remove accents from names
    
    # Step 1: Fetch the page content
    url_east = "https://en.wikipedia.org/wiki/List_of_current_NBA_Eastern_Conference_team_rosters"
    url_west = "https://en.wikipedia.org/wiki/List_of_current_NBA_Western_Conference_team_rosters"

    player_data = []
    for url in [url_east,url_west]:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Step 2: Initialize a list to hold player data

        # Step 3: Find all tables on the page
        tables = soup.find_all('table', class_='toccolours')

        # Step 4: Loop through each table (each team's roster)
        for table in tables:
            # Get the team name from the navbar-header div
            navbar_header = table.find('div', class_='navbar-header')
            if navbar_header:
                team_name = navbar_header.find('b').text.strip()  # Get the team name from the bold text
                team_name = team_name.replace(' roster', '')
                # Step 5: Loop through rows in the current table
                for row in table.find_all('tr')[2:]:  # Skip the header rows
                    cols = row.find_all('td')
                    if cols:
                        player_name = cols[2].text.strip()
                        player_name = player_name.replace('(TW)','')
                        player = player_name.split(', ')
                        if len(player)>1:
                            player_titles = player[1].split(' ')
                            if len(player_titles)==1:
                                player_name = f'{remove_accents(player[1])} {remove_accents(player[0])}'
                            if len(player_titles)>1:
                                player_name=f'{remove_accents(player_titles[0])} {remove_accents(player[0])} {remove_accents(player_titles[1])}'
                        else:
                            player_name = player[0]
                        player_position = cols[0].text.strip()  
                        injured = bool(row.find('span', title="Injured"))  # Set injured parameter

                        player_data.append((player_name, team_name, player_position, injured))

        # Step 6: Create a DataFrame from the collected data
    return pd.DataFrame(player_data, columns=['Name', 'Team', 'Position', 'Injury'])

    
def is_match(name1, name2, threshold=0.8):
    def preprocess_name(name):
        name = name.lower()  # lowercase
        name = re.sub(r'\s+', ' ', name)  # replace multiple spaces with single space
        name = re.sub(r'[^\w\s]', '', name)  # remove punctuation
        name = remove_accents(name) # remove accents
        return name.strip()
    name1 = preprocess_name(name1)
    name2 = preprocess_name(name2)
    score1 = fuzz.ratio(name1, name2) / 100  # Levenshtein-based
    score2 = textdistance.jaccard.similarity(name1, name2)  # Jaccard similarity
    combined_score = (score1 + score2) / 2
    return combined_score >= threshold
