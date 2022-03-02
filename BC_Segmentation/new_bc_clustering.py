#!/usr/bin/env python
# coding: utf-8

# <h1><center>BC Segmentation</center></h1>

# In[1]:


# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

# basic
import os
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Model
import sklearn
from scipy.special import boxcox1p
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, Birch, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('dark')


# In[2]:


bq_client = bigquery.Client()
bq_storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()

sql = """
select *
from coca-cola-datalake-dev.EXT_POI_STAGE.outlet_bc_segment
"""

df_new = bq_client.query(sql, location='EU').to_dataframe(bqstorage_client = bq_storage_client, progress_bar_type='tqdm')


# In[3]:


df_new.info()


# In[4]:


df_new.head()


# In[5]:


df = df_new[["outlet_number","calculated_amount"]]


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


print("Total amount of data: ",len(df_new))


# In[9]:


plt.rcParams["figure.figsize"] = (10,8)
plt.plot(df.calculated_amount)


# #### Normalization

# In[10]:


# Normalize data
def normal_dist(data):
    return (data.apply(lambda x: boxcox1p(x,0.25)))


# In[11]:


df["calculated_amount"].replace(0, 0.000001, inplace=True)


# In[12]:


df["amount_norm"] = normal_dist(df["calculated_amount"])


# In[13]:


# filtering negative amount values (we have *42* negative values here)
df[df['amount_norm']<0] = 0


# In[14]:


model_data = df['amount_norm']


# In[15]:


model_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[16]:


model_data.replace(np.nan, 0, inplace=True)


# In[17]:


len(model_data) # before and after -> 178989 


# In[18]:


model_data.describe()


# In[19]:


sns.displot(model_data)


# #### Model

# In[21]:


## KMeans Model
model = KMeans(n_clusters=3, random_state=42, init='k-means++')
model.fit(model_data.values.reshape(-1, 1))
preds = model.predict(model_data.values.reshape(-1, 1))
df["cluster"] = preds


# In[22]:


sns.scatterplot(data=df, hue='cluster', y='calculated_amount', x='amount_norm', palette=['green','red','dodgerblue'])
plt.title('Cluster Dist.')
plt.show()


# In[23]:


df['cluster'] = df['cluster'].apply(lambda x: 
                                    0 if x==1 # red
                                    else 1 if x==0 # green
                                    else 2 # blue
                                    )


# In[24]:


df["cluster"].value_counts()


# In[25]:


pie=df.groupby('cluster').size().reset_index()
pie.columns=['cluster','value']
px.pie(pie,values='value',names='cluster', title='Cluster Dist.')


# ### Decision Points

# In[26]:


clf = DecisionTreeClassifier(random_state=42)
clf.fit(df['calculated_amount'].values.reshape(-1,1), df['cluster'])
clf_preds = clf.predict(df['calculated_amount'].values.reshape(-1,1))
clf_text = tree.export_text(clf)
print(clf_text)


# ### Results

# In[27]:


final_df = df[["outlet_number","calculated_amount","cluster"]]


# In[28]:


final_df.head()


# In[29]:


low    = final_df.query("cluster == 0").reset_index(drop=True)
middle = final_df.query("cluster == 1").reset_index(drop=True)
high   = final_df.query("cluster == 2").reset_index(drop=True)


# In[30]:


pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[31]:


low.calculated_amount.describe()


# In[32]:


len(low[low['calculated_amount']<0])


# In[33]:


fig = px.box(low, x = "calculated_amount", color_discrete_sequence=["goldenrod"])
fig.show()


# In[34]:


px.histogram(low, x="calculated_amount", color_discrete_sequence=["goldenrod"])


# In[35]:


middle.calculated_amount.describe()


# In[36]:


fig = px.box(middle, x = "calculated_amount", color_discrete_sequence=["darkcyan"])
fig.show()


# In[37]:


px.histogram(middle, x="calculated_amount", color_discrete_sequence=["darkcyan"])


# In[38]:


high.calculated_amount.describe()


# In[39]:


fig = px.histogram(high, x="calculated_amount")
fig.show()


# In[40]:


fig = px.box(high, x = "calculated_amount", color_discrete_sequence=["crimson"])
fig.show()


# In[41]:


px.histogram(high, x="calculated_amount", color_discrete_sequence=["crimson"])


# In[43]:


final_df.to_csv("bc_new_segmentation.csv")


# In[ ]:




