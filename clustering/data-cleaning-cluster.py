#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as  pd


# In[2]:


df = pd.DataFrame(columns=["Player","Diff","KD","Rating"])

file1 = open("clustering/players_kd_rating.txt",encoding='utf8')
lines = file1.read().splitlines()
lines2 = []

for line in lines:
    new_line = line.replace('\t',' ')
    lines2.append(new_line)
    
print(lines2)


# In[3]:


lines2[4]


# In[4]:


for i in range(len(lines2)):
    row = lines2[i].split()
    new_row = {'Player':row[0], 'Diff':row[-3], 'KD':row[-2], 'Rating':row[-1]}
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], axis=0, ignore_index=True)

df


# In[5]:


liste = [x for x in df['Player'] if len(x)<8]
set(liste)


# In[6]:


countries = ['New', 'South','Czech','United','Bosnia','North','Hong']


# In[7]:


for i in range(len(lines2)):
    row = lines2[i].split()
    if row[0] == 'Bosnia':
        new_name = row[0] + row[1] + row[2]
        df['Player'][i] = new_name
    elif row[0] in countries:
        new_name = row[0] + row[1]
        df['Player'][i] = new_name

df


# In[8]:


df.to_csv('clustering/kd_rating.csv', encoding='utf8', index=False)


# In[ ]:




