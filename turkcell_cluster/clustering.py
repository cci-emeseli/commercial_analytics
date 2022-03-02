#!/usr/bin/env python
# coding: utf-8

# <h1><center>Turkcell Traffic </center></h1>

# * Bu çalışmada Turkcell datası, Traditional Channel bazında filtrelenip ardından clustering yapılarak çıktılar alındı.

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
from coca-cola-datalake-dev.EXT_POI_STAGE.outlet_turkcell 
where city_name = "İstanbul"
"""

df_new = bq_client.query(sql, location='EU').to_dataframe(bqstorage_client = bq_storage_client, progress_bar_type='tqdm')


# In[3]:


print("Total amount of data: ", len(df_new))


# In[4]:


print("Total amount of outlet is:", df_new.outlet_number.nunique())


# There are 8664 records for each outlet 

# In[5]:


# Traditional Channel'ı filtreleyelim.
df = df_new.query("main_channel_text == 'TRADITIONAL RETAIL'").reset_index(drop=True)


# In[6]:


df.head(3)


# In[7]:


df.info()


# In[8]:


pivot_table = pd.pivot_table(data=df,index=['outlet_number'], columns=['CALL_HOUR'], values=['BODY_COUNT'])


# In[9]:


pivot_table


# In[10]:


table = pivot_table.reset_index(drop=True)


# In[11]:


# o row a ait saat bazında yoğunluk percentage'ı bulmak istediğimizden row/total yaparak density df oluşturuyoruz.
density_df = table.div(table.sum(axis=1), axis=0)


# In[12]:


density_df


# In[13]:


# Replacing Header with Top Row (Pivot Table'daki BODY_COUNT header'ından kurtulma)
density_df.columns = ["_".join(pair) for pair in density_df.columns]


# In[14]:


density_df.head(3)


# In[15]:


# There is no NaN value
density_df.isna().sum().sum()


# In[16]:


# There is no INF value
density_df.isin([np.inf, -np.inf]).sum().sum()


# In[17]:


# There is no NEGATIVE value
len(density_df[(density_df < 0).all(1)])


# In[18]:


# There is no NULL value
len(density_df[(density_df == 0).all(1)])


# #### Normalization

# In[19]:


'''
# Normalize data
def normal_dist(data):
    return (data.apply(lambda x: boxcox1p(x,0.25)))
'''


# In[20]:


# It's already normalized after creating density dataframe.
sns.displot(density_df.BODY_COUNT_H_00)


# In[21]:


model_data = density_df.copy()


# ## Elbow Method

# In[22]:


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

# In[23]:


hrc_model = AgglomerativeClustering(n_clusters=5, linkage='ward', affinity='euclidean')
hrc_preds = hrc_model.fit_predict(model_data)


# In[24]:


model_data["cluster"] = hrc_preds


# In[25]:


model_data["cluster"].unique()


# In[26]:


model_data['cluster'].value_counts().plot(kind='pie')


# In[136]:


model_data['cluster'].value_counts()


# In[27]:


model_data['cluster'].value_counts()


# ## Decision Points

# In[28]:


from sklearn.model_selection import train_test_split

x = model_data.drop('cluster', axis=1)
y = model_data['cluster']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
dc = DecisionTreeClassifier(criterion="entropy", random_state=42) 
dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)

d_text = tree.export_text(dc)
#print(d_text)


# In[29]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# In[30]:


dc.feature_importances_


# In[172]:


pd.Series(dc.feature_importances_, index=x_train.columns).nlargest(20).plot(kind='barh',figsize=(20, 8))


# ## Results

# ----
# 

# In[174]:


outlet_dens_df = pd.concat([preds_df, density_df], axis=1)


# In[175]:


outlet_dens_df.to_csv("turkcell_hour_cluster.csv")


# -----

# In[39]:


outlet_list = list(pivot_table.index)


# In[41]:


len(outlet_list)


# In[40]:


len(hrc_preds)


# In[54]:


preds_df = pd.DataFrame(hrc_preds, index = outlet_list, columns =['cluster']).reset_index()
preds_df.rename(columns={'index': 'outlet_number'}, inplace=True)


# In[55]:


preds_df


# In[115]:


df["CALL_MONTH"] = df["CALL_MONTH"].replace({
'January' :'01_January',
'February' : '02_February',
'March' : '03_March',
'April' : '04_April', 
'May' :  '05_May',
'June' : '06_June' ,
'July' : '07_July',
'August' : '08_August' ,
'September' : '09_September' ,
'October' : '10_October' ,
'November' : '11_November' ,
'December' : '12_December'})


# In[116]:


result = pd.merge(preds_df, df, on='outlet_number', how='outer')


# In[117]:


result.head()


# In[177]:


result.to_csv("turkcell_all_results.csv")


# In[118]:


cluster_0 = result.query("cluster == 0")
cluster_1 = result.query("cluster == 1")
cluster_2 = result.query("cluster == 2")
cluster_3 = result.query("cluster == 3")
cluster_4 = result.query("cluster == 4")


# <h2><center>General Comparison </center></h2>

# In[147]:


cluster_pie = result.groupby('cluster')['outlet_number'].count().reset_index()

cluster_pie.columns=['cluster','value']
px.pie(cluster_pie, values='value',names='cluster', title="CLUSTER DAĞILIMI")


# In[171]:


result.cluster.value_counts()


# In[134]:


sum_body = result.groupby('cluster')['BODY_COUNT'].sum().reset_index()

sum_body.columns=['cluster','value']
px.pie(sum_body,values='value',names='cluster', title="CLUSTER'LARA GÖRE YOĞUNLUK DAĞILIMI")


# <h2><center> Cluster-0 </center></h2>

# In[120]:


hour_body_0 = cluster_0.groupby('CALL_HOUR')['BODY_COUNT'].sum().reset_index()

fig = px.bar(hour_body_0, x='CALL_HOUR', y='BODY_COUNT',color="BODY_COUNT",
            title= "SAAT BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[122]:


month_body_0 = cluster_0.groupby('CALL_MONTH')['BODY_COUNT'].sum().reset_index()

fig = px.bar(month_body_0, x='CALL_MONTH', y='BODY_COUNT',color="BODY_COUNT",
            title= "AY BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[126]:


pie=cluster_0.groupby('poi_category').size().reset_index()
pie.columns=['poi_category','value']
px.pie(pie,values='value',names='poi_category', title='POI Category Dağılımı')


# In[125]:


pie=cluster_0.groupby('outlet_sub_trade_channel_text').size().reset_index()
pie.columns=['outlet_sub_trade_channel_text','value']
px.pie(pie,values='value',names='outlet_sub_trade_channel_text', title='Outlet Subtrade Channel Dağılımı')


# In[128]:


dist_body_0 = cluster_0.groupby('district_name')['BODY_COUNT'].sum().reset_index()

fig = px.bar(dist_body_0, x='district_name', y='BODY_COUNT',color="BODY_COUNT",
            title= "İLÇELERE GÖRE YOĞUNLUK GRAFİĞİ")
fig.show()


# <h2><center> Cluster-1 </center></h2>

# In[148]:


hour_body_1 = cluster_1.groupby('CALL_HOUR')['BODY_COUNT'].sum().reset_index()

fig = px.bar(hour_body_1, x='CALL_HOUR', y='BODY_COUNT',color="BODY_COUNT",
            title= "SAAT BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[150]:


month_body_1 = cluster_1.groupby('CALL_MONTH')['BODY_COUNT'].sum().reset_index()

fig = px.bar(month_body_1, x='CALL_MONTH', y='BODY_COUNT',color="BODY_COUNT",
            title= "AY BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[151]:


pie=cluster_1.groupby('poi_category').size().reset_index()
pie.columns=['poi_category','value']
px.pie(pie,values='value',names='poi_category', title='POI Category Dağılımı')


# In[152]:


pie=cluster_1.groupby('outlet_sub_trade_channel_text').size().reset_index()
pie.columns=['outlet_sub_trade_channel_text','value']
px.pie(pie,values='value',names='outlet_sub_trade_channel_text', title='Outlet Subtrade Channel Dağılımı')


# In[153]:


dist_body_1 = cluster_1.groupby('district_name')['BODY_COUNT'].sum().reset_index()

fig = px.bar(dist_body_1, x='district_name', y='BODY_COUNT',color="BODY_COUNT",
            title= "İLÇELERE GÖRE YOĞUNLUK GRAFİĞİ")
fig.show()


# <h2><center> Cluster-2 </center></h2>

# In[154]:


hour_body_2 = cluster_2.groupby('CALL_HOUR')['BODY_COUNT'].sum().reset_index()

fig = px.bar(hour_body_2, x='CALL_HOUR', y='BODY_COUNT',color="BODY_COUNT",
            title= "SAAT BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[155]:


month_body_2 = cluster_2.groupby('CALL_MONTH')['BODY_COUNT'].sum().reset_index()

fig = px.bar(month_body_2, x='CALL_MONTH', y='BODY_COUNT',color="BODY_COUNT",
            title= "AY BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[156]:


pie=cluster_2.groupby('poi_category').size().reset_index()
pie.columns=['poi_category','value']
px.pie(pie,values='value',names='poi_category', title='POI Category Dağılımı')


# In[157]:


pie=cluster_2.groupby('outlet_sub_trade_channel_text').size().reset_index()
pie.columns=['outlet_sub_trade_channel_text','value']
px.pie(pie,values='value',names='outlet_sub_trade_channel_text', title='Outlet Subtrade Channel Dağılımı')


# In[158]:


dist_body_2 = cluster_2.groupby('district_name')['BODY_COUNT'].sum().reset_index()

fig = px.bar(dist_body_2, x='district_name', y='BODY_COUNT',color="BODY_COUNT",
            title= "İLÇELERE GÖRE YOĞUNLUK GRAFİĞİ")
fig.show()


# <h2><center> Cluster-3 </center></h2>

# In[159]:


hour_body_3 = cluster_3.groupby('CALL_HOUR')['BODY_COUNT'].sum().reset_index()

fig = px.bar(hour_body_3, x='CALL_HOUR', y='BODY_COUNT',color="BODY_COUNT",
            title= "SAAT BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[160]:


month_body_3 = cluster_3.groupby('CALL_MONTH')['BODY_COUNT'].sum().reset_index()

fig = px.bar(month_body_3, x='CALL_MONTH', y='BODY_COUNT',color="BODY_COUNT",
            title= "AY BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[161]:


pie=cluster_3.groupby('poi_category').size().reset_index()
pie.columns=['poi_category','value']
px.pie(pie,values='value',names='poi_category', title='POI Category Dağılımı')


# In[162]:


pie=cluster_3.groupby('outlet_sub_trade_channel_text').size().reset_index()
pie.columns=['outlet_sub_trade_channel_text','value']
px.pie(pie,values='value',names='outlet_sub_trade_channel_text', title='Outlet Subtrade Channel Dağılımı')


# In[163]:


dist_body_3 = cluster_3.groupby('district_name')['BODY_COUNT'].sum().reset_index()

fig = px.bar(dist_body_3, x='district_name', y='BODY_COUNT',color="BODY_COUNT",
            title= "İLÇELERE GÖRE YOĞUNLUK GRAFİĞİ")
fig.show()


# <h2><center> Cluster-4 </center></h2>

# In[164]:


hour_body_4 = cluster_4.groupby('CALL_HOUR')['BODY_COUNT'].sum().reset_index()

fig = px.bar(hour_body_4, x='CALL_HOUR', y='BODY_COUNT',color="BODY_COUNT",
            title= "SAAT BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[165]:


month_body_4 = cluster_4.groupby('CALL_MONTH')['BODY_COUNT'].sum().reset_index()

fig = px.bar(month_body_4, x='CALL_MONTH', y='BODY_COUNT',color="BODY_COUNT",
            title= "AY BAZLI YOĞUNLUK GRAFİĞİ")
fig.show()


# In[166]:


pie=cluster_4.groupby('poi_category').size().reset_index()
pie.columns=['poi_category','value']
px.pie(pie,values='value',names='poi_category', title='POI Category Dağılımı')


# In[167]:


pie=cluster_4.groupby('outlet_sub_trade_channel_text').size().reset_index()
pie.columns=['outlet_sub_trade_channel_text','value']
px.pie(pie,values='value',names='outlet_sub_trade_channel_text', title='Outlet Subtrade Channel Dağılımı')


# In[168]:


dist_body_4 = cluster_4.groupby('district_name')['BODY_COUNT'].sum().reset_index()

fig = px.bar(dist_body_4, x='district_name', y='BODY_COUNT',color="BODY_COUNT",
            title= "İLÇELERE GÖRE YOĞUNLUK GRAFİĞİ")
fig.show()

