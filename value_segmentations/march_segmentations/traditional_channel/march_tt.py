#!/usr/bin/env python
# coding: utf-8

# <h1><center>Revenue Segmentation of TT</center></h1>

# * Bu notebookta 2021 Mart - 2022 Mart arası dahil olmak üzere bir yıllık data üzerinden Traditional channel'da 4'lü revenue clustering çalışması yapılmıştır.

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


df.isna().sum()


# In[5]:


tt_df = df.query("outlet_main_channel_code == '1'")


# In[6]:


tt_data = tt_df[["outlet_number", "calculated_amount", "round_amount"]]


# In[7]:


tt_data


# In[8]:


len(tt_data) # önceki data 153661


# In[9]:


tt_data.isna().sum()


# In[10]:


print("calculated_amount's min is: ",tt_data.calculated_amount.min())


# In[11]:


print("calculated_amount's max is: ", tt_data.calculated_amount.max())


# In[12]:


print("round_amount's min is: ", tt_data.round_amount.min())


# In[13]:


print("round_amount's max is: ", tt_data.round_amount.max())


# ## Data Prep. for Model

# In[14]:


segm_data = tt_data[["outlet_number", "round_amount"]]


# In[15]:


lower = segm_data[segm_data['round_amount']<1000]
lower["cluster"] = 0
higher = segm_data[segm_data['round_amount']>500000] 
higher["cluster"] = 3


# #### Fiyat sınırı 1000-500.000 TL olarak belirlendiğinde: 

# In[16]:


print("Traditional için 1000 TL altında kalan outlet sayısı:" ,len(lower))


# In[17]:


print("Traditional için 500.000 TL üstünde kalan outlet sayısı:" ,len(higher))


# In[18]:


## filter 1000+ revenue sales.
segm_data = segm_data[segm_data['round_amount']>1000] 
## filter 500.000- revenue sales.
segm_data = segm_data[segm_data['round_amount']<500000] 


# In[19]:


plt.plot(segm_data.round_amount)


# In[20]:


sns.displot(segm_data.round_amount)


# #### Normalization

# In[21]:


segm_data["round_amount"].replace(0, 0.001, inplace=True)


# In[22]:


segm_data['log_amount'] = np.log(segm_data['round_amount'])


# In[23]:


# filtering negative amount values (we have *3* negative values here)
segm_data["log_amount"].clip(lower=0, inplace=True)


# In[24]:


segm_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[25]:


segm_data.replace(np.nan, 0, inplace=True)


# In[26]:


segm_data = segm_data.reset_index(drop=True)


# In[27]:


segm_data.head()


# In[28]:


sns.displot(segm_data.log_amount)


# ## Elbow Method

# In[29]:


model_data = segm_data["log_amount"]


# In[30]:


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


# * It seems quite logical to divide it into clusters 
# <br> of 3 or 4 for the model to be executed.

# #### Model

# In[31]:


## KMeans Model
model = KMeans(n_clusters=4, random_state=42, max_iter=1000000, init='k-means++')
model.fit(model_data.values.reshape(-1, 1))
preds = model.predict(model_data.values.reshape(-1, 1))


# In[32]:


segm_data["cluster"] = preds


# In[33]:


segm_data["cluster"].value_counts()


# In[34]:


segm_data.head(3)


# In[35]:


sns.scatterplot(data=segm_data, hue='cluster', y='round_amount', x='log_amount', palette=['green','red','dodgerblue',"orange"])
plt.title('Cluster Dist.')
plt.show()


# In[37]:


segm_data['cluster'] = segm_data['cluster'].apply(lambda x: 
                                                  0 if x==3 
                                                  else 1 if x==0 
                                                  else 2 if x==2 
                                                  else 3 
                                                  )


# In[38]:


segm_data.head()


# In[39]:


all_data = pd.concat([segm_data, lower, higher]).reset_index(drop=True)


# In[40]:


# higher ve lower kısmı yok burda o yüzden az (609 kadar)
len(segm_data)


# In[41]:


len(all_data)


# In[43]:


len(tt_df)


# In[44]:


pie=all_data.groupby('cluster').size().reset_index()
pie.columns=['cluster','value']
px.pie(pie,values='value',names='cluster', title='Cluster Dist.')


# In[45]:


all_data['cluster'].value_counts()


# In[95]:


len(all_data)


# ### Decision Points

# In[47]:


dc = DecisionTreeClassifier(random_state=42)
dc.fit(all_data['round_amount'].values.reshape(-1,1), all_data['cluster'])
dc_preds = dc.predict(all_data['round_amount'].values.reshape(-1,1))
clf_text = tree.export_text(dc)
print(clf_text)


# Outletlerin cluster'lara göre sınırları ise şu şekildedir.
# - 8805'in altında kalanlar -> Bronze
# - 8805 ve 20578 arasındakiler -> Silver
# - 20578 ve 46930 arasındakiler -> Silver Plus
# - 46930'den büyük olanlar -> Gold 

# In[48]:


fig = px.box(segm_data, x = "round_amount", color_discrete_sequence=["darkcyan"])
fig.show()


# In[87]:


conf_df = pd.merge(all_data, tt_df[["outlet_number","outlet_segment_code"]], on=["outlet_number"]).reset_index(drop=True)
conf_df["cluster"].replace({0:"B", 1: "S", 2:"P", 3:"G"}, inplace=True)


# In[88]:


conf_df.head()


# ### Comparison Matrix

# In[89]:


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

# ------

# ## Results 

# In[91]:


result_df = conf_df[["outlet_number","round_amount","cluster","outlet_segment_code"]]


# In[92]:


result_df.head()


# In[93]:


len(result_df)


# In[94]:


result_df.to_csv("tt_results.csv")

