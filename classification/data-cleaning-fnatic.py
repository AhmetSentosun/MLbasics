#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.DataFrame(columns=["Date","Event","Opponent","Map","Win Round","Delete","Lose Round","Result"])

file1 = open("classification/fnaticmatches.txt",encoding='utf8')
lines = file1.read().splitlines()
lines2 = []

for line in lines:
    new_line = line.replace('\t',' ')
    lines2.append(new_line)
    
print(lines2)


# In[3]:


lines2[1]


# In[4]:


maplist=['Ancient','Anubis','Cache','Cobblestone','Dust2','Dust2_se','Inferno','Inferno_se','KingdomInto','Mirage','Mirage_ce','News','Nuke','Nuke_se','Overpass','Season','Train','Train_se','Vertigo']
new_teams = []
count = 3
while(count < len(lines2)):
    track = lines2[count].split()
    teamname = [i for i in track[:-5] if i not in maplist]
    teamname2 = "".join(teamname)
    print(teamname2)
    new_teams.append(teamname2)
    count += 4



# In[5]:


df = pd.DataFrame(columns=["Date","Event","Opponent","Map","Win Round","Delete","Lose Round","Result"])


# In[6]:


ekstra=lines2[3].split()
ekstra[-5:]


# In[7]:


ekstra=lines2[3].split()
ekstra[-5:]
count = 3
count2 = 0
while count < len(lines2):
    new_row = lines2[count].split()
    new_row2 = new_row[-5:]
    new_row2.append(new_teams[count2])
    lines2[count] = " ".join(new_row2)
    count += 4
    count2 += 1
    


# In[8]:


len(new_teams)


# In[9]:


row = []
for i in range(len(lines2)):
    row.append(lines2[i])
    if len(row) == 4:  
        extras = row[3].split()  
        df=df.append({
            'Date':row[0],
            'Event':row[1],
            'Map':extras[0],
            'Win Round':extras[1],
            'Lose Round':extras[3],
            'Result':extras[4],
        },ignore_index=True)
        row=[]

df['Opponent'] = new_teams
print(df)


# In[10]:


df.tail()


# In[11]:


del df['Delete']


# In[12]:


opps = list(df["Opponent"])
opps_set = set(opps)
print(opps_set)


# In[13]:


tier1_teams = ["SK","ENCE","Gambit","MOUZ", "Vitality","OG","Spirit","fnatic","Astralis","Heroic","Vincere","G2","FaZe","Pyjamas","Cloud9","Liquid","FURIA","Virtus.pro","Renegades","TSM","Titan"]
tier2_teams = ["Complexity","Geniuses", "Eagles", "Riders", "FORZE", "Sprout", "Entropiq", "HAVU", "GODSENT", "HellRaisers", "North", "GamerLegion", "Envy", "MIBR", "Apeks", "SINNERS", "Endpoint", "SAW", "Dignitas"]

print("tier 1 teams:{} \n tier 2 teams: {}".format(len(tier1_teams), len(tier2_teams)))


# In[14]:


tier_list = []
for team in opps:
    lim = len(tier_list)
    t = team.split()
    team_name = t[-1]
    print(team_name)
    for i,j in zip(tier1_teams, tier2_teams):
        if i in team_name:
            tier_list.append(1)
            break
        elif j in team_name:
            tier_list.append(2)
            break

    if len(tier_list) == lim:
        tier_list.append(3)

print(opps)
print(tier_list)


    


# In[15]:


print(len(tier_list))


# In[16]:


df['Opponent Tier'] = tier_list


# In[17]:


df


# In[18]:


set(df['Map'])


# In[19]:


for i in range(len(df['Map'])):
    if df['Map'][i] == 'Train_se':
        df['Map'][i] = 'Train'
    elif df['Map'][i] == 'Dust2_se':
        df['Map'][i] = 'Dust2'
    elif df['Map'][i] == 'Inferno_se':
        df['Map'][i] = 'Inferno'
    elif df['Map'][i] == 'Mirage_ce':
        df['Map'][i] = 'Mirage'
    elif df['Map'][i] == 'Nuke_se':
        df['Map'][i] = 'Nuke'


# In[20]:


df


# In[21]:


for i in range(len(df['Date'])):
    date = df['Date'][i]
    date2 = date.split('/')
    df['Date'][i] = date2[-1]


# In[22]:


df


# In[23]:


for i in range(len(df['Event'])):
    event_name = df['Event'][i]
    for j in range(1,len(event_name)):
        if event_name[-j].isnumeric():
            event_name2 = event_name[:len(event_name)-j]
    df['Event'][i] = event_name2

df


# In[24]:


for i in range(len(df['Result'])):
    if df['Result'][i] == 'T':
        df = df.drop([i])
        


# In[25]:


set(df['Result'])


# In[26]:


df.to_csv('classification/fnatic_matches.csv', encoding='utf-8',index=False)


# In[ ]:




