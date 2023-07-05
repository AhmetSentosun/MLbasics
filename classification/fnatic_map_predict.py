#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('classification/fnatic_matches.csv')
df


# In[3]:


# Veri setinin ilk birkaç satırını gösterme
print(df.head())

# Veri setinin sütunlarını ve veri tiplerini gösterme
print(df.info())

# Sayısal sütunlar için istatistiksel özetleri gösterme
print(df.describe())


# In[4]:


# Seri çizimi ve Korelasyon matrisi için subplotlar oluşturma
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Seri çizimi
sns.scatterplot(data=df, x='Win Round', y='Map', ax=axes[0])
axes[0].set_title('Win Round - Map Seri Çizimi')
axes[0].set_xlabel('Win Round')
axes[0].set_ylabel('Map')

# Korelasyon matrisi
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title('Korelasyon Matrisi')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# In[5]:


# Dağılım grafikleri
plt.figure(figsize=(16, 4))
sns.scatterplot(data=df, x='Map', y='Win Round')
plt.title('Win Round - Map Dağılım Grafiği')
plt.xlabel('Map')
plt.ylabel('Win Round')
plt.show()


# In[6]:


X = df.drop(['Result','Win Round','Lose Round'],axis=1)
y = df['Result']


# In[7]:


column_transform = make_column_transformer((OneHotEncoder(), ['Date','Event','Opponent','Map']),
                                           (OrdinalEncoder(), ['Opponent Tier']))


# In[8]:


X = column_transform.fit_transform(X)


# In[9]:


y = LabelEncoder().fit_transform(y)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=23)


# In[11]:


model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[12]:


from sklearn.svm import SVC

model3 = SVC()
model3.fit(X_train, y_train)


# In[13]:


target_names=set(df['Result'])
target_names


# In[14]:


prediction = model.predict(X_test)
print(classification_report(y_test,prediction,target_names=['Lose','Win']))


# In[15]:


prediction_svc = model3.predict(X_test)
print(classification_report(y_test, prediction_svc, target_names=['Lose', 'Win']))


# In[16]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Decision Tree Classifier
mse_dt = mean_squared_error(y_test, prediction)
rmse_dt = np.sqrt(mse_dt)
accuracy_dt = accuracy_score(y_test, prediction)

# Support Vector Classifier
mse_svc = mean_squared_error(y_test, prediction_svc)
rmse_svc = np.sqrt(mse_svc)
accuracy_svc = accuracy_score(y_test, prediction_svc)

# Karşılaştırma
if mse_dt < mse_svc:
    best_model = 'Decision Tree Classifier'
    best_mse = mse_dt
    best_rmse = rmse_dt
    best_accuracy = accuracy_dt
else:
    best_model = 'Support Vector Classifier'
    best_mse = mse_svc
    best_rmse = rmse_svc
    best_accuracy = accuracy_svc

print("Decision Tree Classifier:")
print(classification_report(y_test, prediction, target_names=['Lose', 'Win']))
print(f"MSE: {mse_dt}")
print(f"RMSE: {rmse_dt}")
print(f"Accuracy: {accuracy_dt}")
print("")

print("Support Vector Classifier:")
print(classification_report(y_test, prediction_svc, target_names=['Lose', 'Win']))
print(f"MSE: {mse_svc}")
print(f"RMSE: {rmse_svc}")
print(f"Accuracy: {accuracy_svc}")
print("")

print(f"En iyi model: {best_model}")
print(f"En düşük MSE: {best_mse}")
print(f"En düşük RMSE: {best_rmse}")
print(f"En yüksek Accuracy: {best_accuracy}")


# In[17]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Decision Tree Classifier
cm_dt = confusion_matrix(y_test, prediction)

# Support Vector Classifier
cm_svc = confusion_matrix(y_test, prediction_svc)

# En iyi model için confusion matrix
if mse_dt < mse_svc:
    best_model_cm = cm_dt
else:
    best_model_cm = cm_svc

# Confusion matrix görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(best_model_cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix - En İyi Model')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()


# In[18]:


feature_importances = model.feature_importances_


encoded_features = column_transform.transformers_[0][1].get_feature_names(['Date', 'Event', 'Opponent', 'Map'])
ordinal_features = ['Opponent Tier']
feature_names = np.concatenate([encoded_features, ordinal_features])

# Öznitelik önem derecelerini sıralama
sorted_indices = np.argsort(feature_importances)[::-1]


for index in sorted_indices:
    print(f"{feature_names[index]}: {feature_importances[index]}")


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[20]:


names = [
    "Nearest_Neighbors",
    "Linear_SVM",
    "RBF_SVM",
    "Decision_Tree",
    "Random_Forest",
    "Neural_Net",
    "AdaBoost",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]


# In[21]:


from sklearn.metrics import f1_score, recall_score, precision_score


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Histogramlar
plt.figure(figsize=(10, 8))
plt.style.use('seaborn')
df.hist(grid=False, edgecolor='k', alpha=0.7)
plt.tight_layout()
plt.show()



# In[23]:


sns.set(style="ticks")
g = sns.pairplot(data=df, hue='Result', plot_kws={"alpha": 0.5})
plt.suptitle('İlişki Grafiği', y=1.02)
plt.show()


# In[24]:


scores = {}

for i in range(len(classifiers)):
    print(i)
    model = classifiers[i]
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)

    scoref1 = f1_score(y_test, prediction)
    scoreprecision = precision_score(y_test,prediction)
    scorerecall = recall_score(y_test, prediction)

    scores[f'{names[i]}'] = [scoref1, scoreprecision, scorerecall]


scores


# In[25]:


#f1 skoruna göre büyükten küçüğe sıralanmış şekilde modellerin metrikleri

for model_name, model_scores in sorted(scores.items(), key=lambda x: x[1][2], reverse=True):
    print(f"Model: {model_name}")
    print(f"F1 Score: {model_scores[0]}")
    print(f"Precision Score: {model_scores[1]}")
    print(f"Recall Score: {model_scores[2]}")
    print()


# In[26]:


keys = [key for key in scores.keys()]
values = [value for value in scores.values()]
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.bar(np.arange(len(keys)) - 0.2, [value[0] for value in values],
       width=0.2, color='b', align='center')
ax.bar(np.arange(len(keys)) + 0,
       [value[1] if len(value) == 3 else 0 for value in values],
       width=0.2, color='g', align='center')
ax.bar(np.arange(len(keys)) + 0.2,
       [value[2] if len(value) == 3 else 0 for value in values],
       width=0.2, color='r', align='center')
ax.set_xticklabels(keys)
ax.set_xticks(np.arange(len(keys)))
ax.legend(['f1 score', 'precision', 'recall'])

plt.show()


# In[27]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0]
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


best_params = grid_search.best_params_


predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("En iyi model: ", best_model)
print("En iyi hiperparametreler: ", best_params)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)


# In[ ]:




