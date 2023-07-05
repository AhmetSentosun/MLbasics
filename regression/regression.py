#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


df = pd.read_csv("regression/xantares_stats.csv")


# Boş ve null değer içeren satırları yazdırma
null_rows = df[df.isnull().any(axis=1)]
empty_rows = df[df.isna().any(axis=1)]

print("Boş değer içeren satırlar:")
print(empty_rows)

print("\nNull değer içeren satırlar:")
print(null_rows)


# In[2]:


#unique values
unique_values = sorted(set(df["Team"]).union(set(df["Opponent"])).union(set(df["Map"])).union(set(df["Date"])))


value_map = {value: index+1 for index, value in enumerate(unique_values)}


df["Team"] = df["Team"].apply(lambda x: value_map[x])
df["Opponent"] = df["Opponent"].apply(lambda x: value_map[x])
df["Map"] = df["Map"].apply(lambda x: value_map[x])

df["Date"]=df["Date"].apply(lambda x: x.split("/")[2])


df.to_csv("regression/news_file.csv", index=False)


# In[3]:


import pandas as pd
df = pd.read_csv("news_file.csv")


# In[4]:


df['Date']


# In[5]:


print(df.info)


# In[6]:


df.head()


# In[7]:


df.tail()


# # visualization

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(12, 18))

for i, column in enumerate(df.columns):
    sns.histplot(data=df, x=column, ax=axes[i], kde=True)
    axes[i].set_title(column)
    axes[i].set_xlabel(column)
    

plt.tight_layout()
plt.show()


# In[10]:


# Korelasyon Tablosu
correlation_table = df.corr()
print(correlation_table)


# In[11]:


# Scatter Plot
plt.scatter(df['Map'], df['Rating'])
plt.xlabel('Map')
plt.ylabel('Ranting')
plt.title('Scatter Plot')
plt.show()


# In[12]:


df["Date"].value_counts()


# In[13]:


date_counts = df["Date"].value_counts()

# Pasta grafiği
plt.pie(date_counts, labels=date_counts.index, autopct='%1.1f%%')
plt.title('Maçların Tarih Dağılımı')
plt.axis('equal')
plt.show()


# In[14]:


df["Team"].value_counts()


# In[15]:


df["Opponent"].value_counts()


# In[16]:


map_counts=df["Map"].value_counts()



# Pasta grafiği
plt.pie(map_counts, labels=map_counts.index, autopct='%1.1f%%')
plt.title('Maçların Harita Dağılımı')
plt.axis('equal')
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
X = df.drop('Rating',axis=1)
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[18]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[19]:


#decisionTree model
decision_treeReg=DecisionTreeRegressor()
decision_treeReg.fit(X_train,y_train)


# In[20]:


rm = RandomForestRegressor()
rm.fit(X_train,y_train)


# In[21]:


X_train


# In[22]:


predictionsDecision_treeReg = decision_treeReg.predict(X_test)
pred_real = pd.DataFrame({'Gerçek Değer': y_test, 'Tahmin': predictionsDecision_treeReg})

# İlk 30
print(pred_real.head(30))


# In[23]:


predictions = rm.predict(X_test)
pred_real = pd.DataFrame({'Gerçek Değer': y_test, 'Tahmin': predictions})

# İlk 30
print(pred_real.head(30))


# In[24]:


# Karar ağacı regresyonunda
feature_importance_dt = pd.DataFrame({'Feature': X.columns, 'Importance': decision_treeReg.feature_importances_})
features_dt = feature_importance_dt.nlargest(20, 'Importance')
print("Karar Ağacı Regresyonunda Özellikler:")
print(features_dt)

# Rastgele orman regresyonunda
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': rm.feature_importances_})
features_rf = feature_importance_rf.nlargest(20, 'Importance')
print("\nRastgele Orman Regresyonunda Özellikler:")
print(features_rf)


# In[25]:


#plt.scatter(y_test, predictions): Gerçek değerler (y_test) ile tahmin edilen değerler (predictions) arasındaki 
#ilişkiyi gösteren scatter plot grafiği

import matplotlib.pyplot as plt

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predictions')


# In[26]:


from sklearn import metrics
import numpy as np

metrics_decision_treeReg = {
    'MAE': metrics.mean_absolute_error(y_test, predictionsDecision_treeReg),
    'MSE': metrics.mean_squared_error(y_test, predictionsDecision_treeReg),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, predictionsDecision_treeReg)),
    'EVS': metrics.explained_variance_score(y_test, predictionsDecision_treeReg),
    'R2 Score': metrics.r2_score(y_test, predictionsDecision_treeReg)
}

metrics_random = {
    'MAE': metrics.mean_absolute_error(y_test, predictions),
    'MSE': metrics.mean_squared_error(y_test, predictions),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    'EVS': metrics.explained_variance_score(y_test, predictions),
    'R2 Score': metrics.r2_score(y_test, predictions)
}

df_comparison = pd.DataFrame({'Decision Tree Regressor': metrics_decision_treeReg,
                              'RandomForest Regressor ': metrics_random})

# Tabloyu yazdırma
print(df_comparison)


# In[27]:


import pandas as pd
pred_table = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': predictions.astype(float)})

pred_table


# In[28]:


df.iloc[0]


# In[29]:


import seaborn as sns

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,4))
sns.histplot(data=pred_table, x='Actual', bins=30, kde=True, ax=ax1)
ax1.set_title('Diff Histogram')
sns.histplot(data=pred_table, x='Predicted', bins=30, kde=True, ax=ax2)
ax2.set_title('ABS Histogram')
plt.show()


# In[30]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}


rm = RandomForestRegressor()

#grid search to find the best hyperp.
grid_search = GridSearchCV(estimator=rm, param_grid=param_grid, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best model and its predictions
best_rm = grid_search.best_estimator_
predictions = best_rm.predict(X_test)


mae = mean_absolute_error(y_test, predictions)


print("Best Hyperparameters:", grid_search.best_params_)
print("MAE:", mae)

