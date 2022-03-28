#!/usr/bin/env python
# coding: utf-8

# <h1><center>Revenue Segmentation of OP</center></h1>

# * Bu notebookta 2021 Mart - 2022 Mart arası dahil olmak üzere bir yıllık data üzerinden OP channel'da 3'lü revenue clustering çalışması yapılmıştır.

# In[1]:


# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

# Basic
import os
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import datetime as dt

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Model
import sklearn
from scipy.special import boxcox1p
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[2]:


bq_client = bigquery.Client()
bq_storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()

sql = """
select *	
from EXT_POI_STAGE.outlet_revenue_segment
"""

df = bq_client.query(sql, location='EU').to_dataframe(bqstorage_client = bq_storage_client, progress_bar_type='tqdm')


# In[3]:


df.head()


# In[4]:


op_df = df.query("outlet_main_channel_code == '2'")
#1-> TT
#2-> OP


# In[5]:


op_data = op_df[["outlet_number", "calculated_amount", "round_amount"]]


# In[6]:


op_data.head()


# In[7]:


len(op_data)


# In[8]:


op_data.isnull().sum()


# In[9]:


print("calculated_amount's min is: ",op_data.calculated_amount.min())


# In[10]:


print("calculated_amount's max is: ", op_data.calculated_amount.max())


# In[11]:


print("round_amount's min is: ", op_data.round_amount.min())


# In[12]:


print("round_amount's max is: ", op_data.round_amount.max())


# ## Data Prep. for Model

# In[13]:


segm_data = op_data[["outlet_number", "round_amount"]]


# In[14]:


lower = segm_data[segm_data['round_amount']<100]
lower["cluster"] = 0
higher = segm_data[segm_data['round_amount']>500000] 
higher["cluster"] = 2


# #### Fiyat sınırı 100-500000 TL olarak belirlendiğinde: 

# In[15]:


print("OP için 100 TL altında kalan outlet sayısı:" ,len(lower))


# In[16]:


print("OP için 500.000 TL üstünde kalan outlet sayısı:" ,len(higher))


# In[17]:


## filter 1000+ revenue sales.
segm_data = segm_data[segm_data['round_amount']>100] 
## filter 500.000- revenue sales.
segm_data = segm_data[segm_data['round_amount']<500000] 


# In[18]:


plt.plot(segm_data.round_amount)


# In[19]:


sns.displot(segm_data.round_amount)


# #### Normalization

# In[20]:


segm_data["round_amount"].replace(0, 0.001, inplace=True)


# In[21]:


segm_data['log_amount'] = np.log(segm_data['round_amount'])


# In[22]:


# filtering negative amount values (we have *3* negative values here)
segm_data["log_amount"].clip(lower=0, inplace=True)


# In[23]:


segm_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[24]:


segm_data.replace(np.nan, 0, inplace=True)


# In[25]:


segm_data = segm_data.reset_index(drop=True)


# In[26]:


segm_data.head()


# In[27]:


sns.displot(segm_data.log_amount)


# ## Elbow Method

# In[28]:


model_data = segm_data["log_amount"]


# In[29]:


Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(model_data.values.reshape(-1, 1))
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# ## Model

# In[30]:


## KMeans Model
model = KMeans(n_clusters=3, random_state=42, max_iter=1000000, init='k-means++')
model.fit(model_data.values.reshape(-1, 1))
preds = model.predict(model_data.values.reshape(-1, 1))


# In[31]:


segm_data["cluster"] = preds


# In[32]:


segm_data["cluster"].value_counts()


# In[33]:


segm_data.head(3)


# In[34]:


sns.scatterplot(data=segm_data, hue='cluster', y='round_amount', x='log_amount', palette=['green','red','dodgerblue'])
plt.title('Cluster Dist.')
plt.show()


# In[36]:


segm_data['cluster'] = segm_data['cluster'].apply(lambda x: 
                                                  0 if x==0
                                                  else 1 if x==2
                                                  else 2 
                                                  )


# In[37]:


all_data = pd.concat([segm_data, lower, higher]).reset_index(drop=True)


# In[38]:


len(all_data)


# In[39]:


len(op_df)


# In[40]:


pie=all_data.groupby('cluster').size().reset_index()
pie.columns=['cluster','value']
px.pie(pie,values='value',names='cluster', title='Cluster Dist.')


# In[41]:


all_data['cluster'].value_counts()


# ### Decision Points

# In[42]:


dc = DecisionTreeClassifier(random_state=42)
dc.fit(all_data['round_amount'].values.reshape(-1,1), all_data['cluster'])
dc_preds = dc.predict(all_data['round_amount'].values.reshape(-1,1))
clf_text = tree.export_text(dc)
print(clf_text)


# Outletlerin cluster'lara göre sınırları ise şu şekildedir.
# - 4990'nın altında kalanlar -> Bronze
# - 4990 ve 24206 arasındakiler -> Silver
# - 24206'den büyük olanlar -> Gold 

# In[47]:


conf_df = pd.merge(all_data, op_df[["outlet_number","outlet_segment_code"]], on=["outlet_number"]).reset_index(drop=True)
conf_df["cluster"].replace({0:"B", 1: "S", 2:"G"}, inplace=True)


# In[48]:


conf_df.head()


# ### Confusion Matrix

# In[49]:


d1 = pd.pivot_table(data=conf_df.reset_index()
              , index='cluster'
              , columns='outlet_segment_code'
              , values='outlet_number'
              , aggfunc=np.size, sort=True
              , fill_value=0).sort_index()

d1 = d1[sorted(d1.columns)]
print(d1)
sns.heatmap(d1, center=0).set_title('Model')
plt.show()


# - Model sonucunda çıkan segment bilgileri ile mevcut 
# <br> durumdaki segment bilgisinin outlet bazında sayısal
# <br> olarak nasıl değiştiği yukarıdaki gibidir.

# ## Results 

# In[54]:


results = conf_df[["outlet_number","round_amount","cluster","outlet_segment_code"]].reset_index(drop=True)


# In[55]:


results.head()


# In[56]:


len(results)


# In[ ]:


results.to_csv("op_results.csv")

