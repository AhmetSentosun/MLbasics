#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df = pd.read_csv('clustering/kd_rating.csv')
df.head(10)


# In[3]:


df.tail()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('clustering/kd_rating.csv')

# İki grafik için subplotlar oluşturma
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Grafik 1: Rating Dağılımı için Histogram
axes[0].hist(df['Rating'], bins=10, edgecolor='black')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Frekans')
axes[0].set_title('Rating Dağılımı')

# Grafik 2: KD Değerlerinin Kutu Grafiği
axes[1].boxplot(df['KD'])
axes[1].set_ylabel('KD')
axes[1].set_title('KD Değerlerinin Kutu Grafiği')

# Grafiklerin sıkışıklığı düzenleme
plt.tight_layout()

# Grafikleri gösterme
plt.show()


# In[4]:


plt.figure(figsize=(12, 5))
plt.scatter(df['Diff'], df['Rating'], alpha=0.5)
plt.xlabel('Diff')
plt.ylabel('Rating')
plt.title('Diff ve Rating Arasındaki İlişki')
plt.show()


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


X = df.drop(['Player','Diff'],axis=1)
X


# In[9]:


from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[10]:


from sklearn.cluster import AgglomerativeClustering

n_clusters = 5  
agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
agglomerative.fit(X)


# In[11]:


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# In[12]:


kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
df['cluster']= kmeansmodel.fit_predict(X)


# In[13]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

n_clusters = 5
agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
agglomerative.fit(X)


silhouette = silhouette_score(X, agglomerative.labels_)
calinski_harabasz = calinski_harabasz_score(X, agglomerative.labels_)

print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Score:", calinski_harabasz)


# In[14]:


#Küme içi yayılma (Inertia): Küme içindeki noktaların merkeze olan uzaklıklarının karelerinin toplamıdır.
#Silhouette skoru: Bu skor, bir veri noktasının kendi kümesine olan benzerliği ile diğer kümelere olan farklılıklarını ölçer.
#Calinski-Harabasz skoru: Bu skor, küme içi yayılmayı küme arasındaki yayılmaya oranlayarak bir kümeleme performans ölçüsüdür.

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


silhouette_score = silhouette_score(X, kmeansmodel.labels_)
calinski_harabasz_score = calinski_harabasz_score(X, kmeansmodel.labels_)


print("Silhouette Score:", silhouette_score)
print("Calinski-Harabasz Score:", calinski_harabasz_score)


# In[15]:


centroids = kmeansmodel.cluster_centers_
cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]
df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

colors = ['#DF2020', '#81DF20', '#2095DF']
df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

plt.scatter(df.KD, df.Rating, c=df.c, alpha = 0.6, s=10)


# In[16]:


from matplotlib.lines import Line2D
fig, ax = plt.subplots(1, figsize=(8,8))
# plot data
plt.scatter(df.KD, df.Rating, c=df.c, alpha = 0.6, s=10)
# plot centroids
plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)
# plot KD mean
plt.plot([df.KD.mean()]*2, [0.8,1.4], color='black', lw=0.5, linestyle='--')
plt.xlim(0.8,1.4)
# plot Rating mean
plt.plot([0.8,1.4], [df.Rating.mean()]*2, color='black', lw=0.5, linestyle='--')
plt.ylim(0.8,1.4)
# create a list of legend elemntes
## average line
legend_elements = [Line2D([0], [0], color='black', lw=0.5, linestyle='--', label='Average')]
## markers / records
cluster_leg = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
               markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
## centroids
cent_leg = [Line2D([0], [0], marker='^', color='w', label='Centroid - C{}'.format(i+1), 
            markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)]
# add all elements to the same list
legend_elements.extend(cluster_leg)
legend_elements.extend(cent_leg)
# plot legend
plt.legend(handles=legend_elements, loc='upper right', ncol=2)
# title and labels
plt.title('Player\n', loc='left', fontsize=22)
plt.xlabel('KD')
plt.ylabel('Rating')


# In[17]:


import numpy as np

#küme merkezi
centroids = kmeansmodel.cluster_centers_

distances = []
for i in range(len(centroids)):
    centroid = centroids[i]
    distance = np.linalg.norm(X - centroid, axis=1)
    distances.append(distance)


feature_importance = np.mean(distances, axis=0)

sorted_indices = np.argsort(feature_importance)[::-1]


for i in sorted_indices:
    print("Feature", i+1, "- Importance:", feature_importance[i])


# In[18]:


# Cross-correlation table
correlation_matrix = df.corr()
print(correlation_matrix)


# In[19]:


plt.figure(figsize=(4, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[20]:


print(df.describe())


# In[ ]:




