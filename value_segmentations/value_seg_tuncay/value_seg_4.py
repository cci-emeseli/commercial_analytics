#!/usr/bin/env python
# coding: utf-8

# In[53]:


import os

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import IsolationForest

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('bright')


# In[2]:


data = pd.read_excel('Outlet_Rev_Kar 26.08.21.xlsx', sheet_name='Sheet1')


# In[3]:


data.head()


# In[4]:


data['Outlet Number'] = data['Outlet Number'].astype(object)


# In[5]:


data.info()


# In[6]:


data.describe()


# In[8]:


plt.figure(figsize=(10, 8))
sns.scatterplot(data=data, hue='Segment', x='Revenue', y='Kar')
plt.show()


# In[14]:


sns.displot(data['Revenue'])


# In[15]:


sns.displot(data['Kar'])


# ---
# 
# ## Cluster

# In[9]:


## set. index Outlet Number

data = data.set_index('Outlet Number')


# In[26]:


model_data_1 = data[['Revenue', 'Kar']]


# In[27]:


Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(model_data_1)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# In[28]:


model_1 = KMeans(n_clusters=4, random_state=42, max_iter=100000, init='random')
model_1.fit(model_data_1)
preds = model_1.predict(model_data_1)

# model_data_1 = pd.concat([model_data_1, pd.DataFrame({'preds':preds}, index=model_data_1.index)], axis=1)
model_data_1.loc[:, 'preds'] = preds

result_1 = pd.merge(model_data_1, data['Segment'], how='inner', left_index=True, right_index=True)
result_1['preds'] = result_1['preds'].astype('category')
plt.figure(figsize=(10, 8))
sns.scatterplot(data=result_1.reset_index(), hue='preds', y='Revenue', x='Kar')
plt.title('Model 1 - Revenue')
plt.show()


# In[29]:


result_1['preds'].value_counts()


# In[30]:


data[['Revenue', 'Kar']].corr()


# In[31]:


data['log_revenue'] = np.log(data['Revenue'])
data['log_kar'] = np.log(data['Kar'])


# In[32]:


sns.displot(data['log_revenue'])


# In[52]:


sns.displot(np.log(np.log(data['log_kar'])))


# ---
# 
# ## Log Cluster

# In[69]:


model_data_2 = data[['log_revenue', 'log_kar']]


# In[35]:


Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(model_data_2)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# In[70]:


model_2 = KMeans(n_clusters=2, random_state=42, max_iter=100000, init='random')
model_2.fit(model_data_2)
preds = model_1.predict(model_data_2)

# model_data_1 = pd.concat([model_data_1, pd.DataFrame({'preds':preds}, index=model_data_1.index)], axis=1)
model_data_2.loc[:, 'preds'] = preds

result_2 = pd.merge(model_data_2, data['Segment'], how='inner', left_index=True, right_index=True)
result_2['preds'] = result_2['preds'].astype('category')
plt.figure(figsize=(10, 8))
sns.scatterplot(data=result_2.reset_index(), hue='preds', y='log_revenue', x='log_kar')
plt.title('Model 2 - Revenue')
plt.show()


# In[72]:


data.head()


# In[74]:


data.reset_index().to_csv('data_20210826.csv', sep='|'
                          , columns=['Outlet Number', 'Segment', 'Revenue', 'Kar']
                         , index=False)


# In[ ]:




