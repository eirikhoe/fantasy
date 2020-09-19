import pandas as pd
import numpy as np
import re


path  = 'C:/Users/Eirik/Documents/Python_Scripts/Fantasy/fantasy_historikk'

text_regex = re.compile(r'((?:\S+ ?)+)')
team_regex = re.compile(r'\((\d)\) ((?: ?\S+)+)') 
score_regex = re.compile(r'(\d{1,3}(?:\.\d{2}){0,1})')
playoffs = []
for year in range(2008,2019):

    with open(f'{path}/fantasy_{year}_playoffs.txt') as f:
            lines = f.readlines()

    linetypes = {0:text_regex,1:team_regex,2:score_regex,3:team_regex,4:score_regex}
    count = 0
    res = []
    games = []

    for line in lines:
        text = text_regex.search(line)
        if text:
            # Handle special case of bye week for a team.
            if (text.group() == 'Bye') and (count == 2):
                # game = [res[0],res[1][0],res[1][1],np.NaN,0,'Bye',np.NaN]
                res = []
                count = 0
                continue
            
            res.append(linetypes[count].findall(line)[0])    
            count =  (count + 1) % 5
            
            # Handle regular case of a game.
            if count == 0:
                game = [res[0],int(res[1][0]),res[1][1],float(res[2]),int(res[3][0]),res[3][1],float(res[4])]
                res = []
                games.append(game)
            

    playoff = pd.DataFrame(games,columns = ['Game Type','First Team Rank', 'First Team Name', 'First Team Pts', 'Second Team Rank', 'Second Team Name', 'Second Team Pts'])
    playoff['Year'] = year
    playoffs.append(playoff)

playoffs = pd.concat(playoffs,axis=0,sort=False,ignore_index=True)

# Fix consolation semifinal being called semifinal
playoffs.loc[(playoffs['Game Type']=='Semifinal') 
             & (playoffs['First Team Rank']>4),
            'Game Type'] = 'Consolation Semifinal'

switched_playoffs = playoffs.iloc[:,[0,4,5,6,1,2,3,7]]

switched_playoffs = switched_playoffs.rename(columns=dict(zip(switched_playoffs.columns.values,playoffs.columns.values)))
temp = playoffs.append(switched_playoffs)
temp.iloc[0::2,:] = playoffs.values
temp.iloc[1::2,:] = switched_playoffs.values

playoffs = temp.iloc[:,[7,0,1,2,3,4,5,6]]
playoffs = playoffs.rename(columns=dict(zip(playoffs.columns,['Year','Game','Team Rank','Team','Pts For','Opponent Rank','Opponent','Pts Against'])))
playoffs['Win'] = (playoffs['Pts For'] > playoffs['Pts Against']).astype(int)