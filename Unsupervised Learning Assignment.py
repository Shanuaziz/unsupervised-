#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data=pd.read_csv('Wine_clust .csv')


# In[7]:


data.head()


# In[9]:


data.info()


# In[10]:


data.columns


# In[12]:


data.isnull().sum()


# In[13]:


sns.scatterplot(x='Alcohol',y='Color_Intensity',data=data)


# In[14]:


#K clustering


# In[15]:


X=data
K=3


# In[16]:


Centroids =(X.sample(n=K))
plt.scatter(X['Alcohol'],X['Color_Intensity'],c='black')
plt.scatter(Centroids['Alcohol'],Centroids['Color_Intensity'],c='red')
plt.xlabel('Alcohol')
plt.ylabel('Color_Intensity')
plt.show()


# In[17]:


from sklearn.cluster import KMeans


# In[18]:


K=3
kmeans=KMeans(n_clusters=K,random_state=0)
X['Cluster']=kmeans.fit_predict(X[['Alcohol','Color_Intensity']])


# In[19]:


X.head(10)


# In[20]:


centroids=kmeans.cluster_centers_


# In[21]:


sns.scatterplot(data=X, x='Alcohol',y='Color_Intensity', hue='Cluster', palette='Set1')
plt.scatter(Centroids[:, 0],Centroids[:, 1],c='red', marker='x', s=100)
plt.xlabel('Alcohol')
plt.ylabel('Color_Intensity')
plt.show()


# In[22]:


inertia=kmeans.inertia_
print('Inertia:', inertia)


# In[23]:


K_range=range(1,11)
inertia_values=[]


# In[24]:


for K in K_range:
    kmeans=KMeans(n_clusters=K,random_state=0)
    kmeans.fit(X[['Alcohol','Color_Intensity']])
    inertia=kmeans.inertia_
    inertia_values.append(inertia)


# In[25]:


plt.plot(K_range,inertia_values,marker='o',c='g')
plt.title('Elboe Method for Optimal K')
plt.xlabel('Number of Clusters(K)')
plt.ylabel('Inertia')
plt.show()


# In[28]:


#Evaluating the algorithm


# In[29]:


from sklearn.metrics import silhouette_score


# In[30]:


silhouette_avg=silhouette_score(X[['Alcohol','Color_Intensity']],X['Cluster'])
print('Silhouette Score:',silhouette_avg)


# Hierarchical clustering 

# In[32]:


data.head()


# In[35]:


X= data.iloc[:,[0,9]]
X.head()


# In[36]:


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


# In[73]:


dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Wine')
plt.ylabel('Euclidean distances')
plt.show()


# In[74]:


from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)


# In[75]:


y_hc


# In[76]:


X_np=X.values


# In[77]:


import matplotlib.pyplot as plt


# In[42]:


for cluster_label in range(5):   
    plt.scatter(X_np[y_hc==cluster_label,0],X_np[y_hc== cluster_label, 1],s=100,label=f'Cluster{cluster_label}')
    
plt.title('Clusters of wine')
plt.xlabel('Alcohol')
plt.ylabel('Color_Intensity')
plt.legend()
plt.show()


# DBSCAN

# In[45]:


data.head()


# In[46]:


data.info()


# In[47]:


x = data.values


# In[48]:


x.shape


# In[79]:


X= data.iloc[:,[0,9]]
X.head()


# In[80]:


plt.scatter(x[:,0],x[:,1])
plt.xlabel('Alcohol')
plt.ylabel('Color_Intensity')
plt.show()


# In[81]:


from sklearn.cluster import DBSCAN


# In[86]:


dbscan = DBSCAN(eps=0.3,min_samples=2)


# In[87]:


labels=dbscan.fit_predict(data)


# In[88]:


np.unique(labels)


# In[ ]:




