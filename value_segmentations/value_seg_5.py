#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd

from google.cloud import bigquery

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, Birch, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('bright')


# In[2]:


bq_client = bigquery.Client()


# In[89]:


sql = """
select rk.*
       , ifnull(d.avg_door_number , 0) as avg_door_number
from  `predictive_order.tmp_revenue_kar` rk 
      left join 
      (
        select d.outlet_number 
               , avg(d.door_number) as avg_door_number
        from  `cooler_modeling_eu.tbl_door_numbers` d 
        where d._HistoryDate >= '2020-08-01'
              and d._HistoryDate < '2021-08-01'
              and d.CountryCode = 'TR'
        group by d.outlet_number 
      ) d on d.outlet_number = cast(rk.Outlet_Number as string)
"""

data_2 = bq_client.query(query=sql).to_dataframe()


# In[90]:


data_2 = data_2.set_index('Outlet_Number')


# In[23]:


data = pd.read_excel('Outlet_Rev_Kar 26.08.21.xlsx', sheet_name='Sheet1')


# In[26]:


data['Outlet_Number'] = data['Outlet Number'].astype(object)


# In[27]:


data.head()


# In[28]:


data = data.drop('Outlet Number', axis=1)


# In[29]:


data.describe()


# In[30]:


data.head()


# In[31]:


## set. index Outlet Number

data = data.set_index('Outlet_Number')


# ---
# 
# ## Revenue Cluster

# In[32]:


data.head()


# In[33]:


data['log_revenue'] = np.log(data['Revenue'])
data['log_kar'] = np.log(data['Kar'])


# In[34]:


sns.displot(data['Revenue'])


# In[35]:


sns.displot(data['log_revenue'])


# In[53]:


rev_model_data = data['log_revenue']


# In[43]:


Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(rev_model_data.values.reshape(-1, 1))
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


model_rev = KMeans(n_clusters=4, random_state=42, max_iter=1000000, init='k-means++')
model_rev.fit(rev_model_data.values.reshape(-1, 1))
preds = model_rev.predict(rev_model_data.values.reshape(-1, 1))

rev_model_data = pd.concat([rev_model_data
                            , pd.DataFrame({'preds':preds}
                                           , index=rev_model_data.index)], axis=1)

result_1 = pd.merge(rev_model_data, data[['Segment', 'log_kar']], how='inner', left_index=True, right_index=True)
result_1['preds'] = result_1['preds'].astype('category')
# plt.figure(figsize=(15, 15))
sns.scatterplot(data=result_1.reset_index(), hue='preds', y='log_revenue', x='log_kar')
plt.title('Model 1 - Revenue')
plt.show()


# In[55]:


result_1['org_revenue'] = np.exp(result_1['log_revenue'])
result_1['org_kar'] = np.exp(result_1['log_kar'])


# In[56]:


result_1['preds'].value_counts()


# In[57]:


sns.scatterplot(data=result_1.reset_index(), hue='preds', y='org_revenue', x='org_kar')
plt.title('Model 1 - Revenue')
plt.show()


# In[58]:


msk = result_1['preds'] != 0
sns.scatterplot(data=result_1[msk].reset_index(), hue='preds', y='org_revenue', x='org_kar')
plt.title('Model 1 - Revenue')
plt.show()


# In[59]:


result_1['rank'] = result_1['preds'].apply(lambda x: 
                                           0 if x==1 # orange
                                           else 1 if x==2 # green
                                           else 2 if x==3 # red
                                           else 3
                                          )


# In[67]:


result_1['rank'].value_counts()


# In[60]:


clf = DecisionTreeClassifier(random_state=42)
clf.fit(result_1['org_revenue'].values.reshape(-1,1), result_1['rank'])
clf_preds = clf.predict(result_1['org_revenue'].values.reshape(-1,1))
clf_text = tree.export_text(clf)<
print(clf_text)


# In[61]:


fig = plt.figure(figsize=(10, 10))

_ = tree.plot_tree(clf
                 , feature_names=['Revenue']
                 , filled=True)


# In[62]:


d1 = pd.pivot_table(data=result_1.reset_index()
              , index='Segment'
              , columns='rank'
              , values='Outlet_Number'
              , aggfunc=np.size, sort=True
              , fill_value=0).sort_index()

d1 = d1[sorted(d1.columns)]
print(d1)
sns.heatmap(
            d1
            , center=0
    
).set_title('Model')

plt.show()


# In[88]:


d1 = pd.pivot_table(data=result_2.reset_index()
              , index='new_segment'
              , columns='rank'
              , values='Outlet_Number'
              , aggfunc=np.size, sort=True
              , fill_value=0).sort_index()

d1 = d1[sorted(d1.columns)]
print(d1)
sns.heatmap(
            d1
            , center=0
    
).set_title('Model')

plt.show()


# In[70]:


result_1['rank'].value_counts()


# In[68]:


result_1.head()


# In[76]:


data['new_segment'] = data['Revenue'].apply(lambda x: 'Bronze' if (x >= 0 and x< 11520)
                                           else 'Silver' if (x>=11520) and (x<43200)
                                           else 'Gold')


# In[85]:


result_1.head()


# In[86]:


result_2  = pd.merge(result_1, data['new_segment'], left_index=True, right_index=True)


# In[77]:


data['new_segment'].value_counts()


# In[78]:


data['Segment'].value_counts()


# In[81]:


result_1.groupby('rank', as_index=False).agg({'org_revenue':np.mean})


# In[82]:


result_1.groupby('rank', as_index=False).agg({'org_revenue':np.median})


# In[83]:


result_1.groupby('rank', as_index=False).agg({'org_revenue':np.min})


# In[84]:


result_1.groupby('rank', as_index=False).agg({'org_revenue':np.max})


# In[92]:


data_2.head()


# In[93]:


result_2 = pd.merge(result_2, data_2['avg_door_number'], left_index=True, right_index=True)


# In[94]:


result_2.head()


# In[96]:


result_2.groupby('rank', as_index=False).agg({'avg_door_number': np.mean})


# In[97]:


result_2.groupby('new_segment', as_index=False).agg({'avg_door_number': np.mean})


# In[98]:


result_2.groupby('Segment', as_index=False).agg({'avg_door_number': np.mean})


# In[99]:


result_2.to_csv('results_20210831.csv', sep='|')

