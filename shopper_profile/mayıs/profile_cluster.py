#!/usr/bin/env python
# coding: utf-8

# <h1><center>Shopper Profile</center></h1>

# * Bu notebook Traditional Channel'da İstanbuldaki shopper profile'ları bulmak için hier. clustering'in kullanıldığı bir çalışmadır. 
# 

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
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('dark')


# In[2]:


bq_client = bigquery.Client()
bq_storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()

sql = """
select *
from EXT_POI_STAGE.SHOPPER_RB_SALES_TR
"""

df = bq_client.query(sql, location='EU').to_dataframe(bqstorage_client = bq_storage_client, progress_bar_type='tqdm')


# In[3]:


df.head()


# In[4]:


list(df.columns)


# ## EDA

# In[5]:


df.info() 


# In[6]:


df.MAIN_CHANNEL.unique()


# In[7]:


df.outlet_number.nunique()


# In[8]:


df['outlet_number'] = df['outlet_number'].astype(object)


# In[9]:


df = df.query("ILADI == 'İstanbul'").reset_index(drop=True)


# In[10]:


df = df.query("MAIN_CHANNEL == 'TRADITIONAL RETAIL'").reset_index(drop=True)


# In[11]:


# Bu dataframe üstünde oynama yapılmamış ana datadır.
df.outlet_number.nunique()


# In[12]:


# bu dataframe, model için kullanılacak dataframe'dir
model_data = df.copy()


# In[13]:


# 'GIDAVEICECEK_ORAN', 'LOKANTAOTEL_ORAN',
model_data = model_data[[      
'SEGMENT',
'outlet_ses',
'age_cluster',
'YAYA_TRAFIGI',
'AYLIK_HARCAMA',
'GIDAVEICECEK',
'HANE_BUYUKLUGU',
'ALKOLTUTUN',
'EGLENCEKULTUR',
'LOKANTAOTEL',
'ALKOLTUTUN_ORAN',
'EGLENCEKULTUR_ORAN',
'GIDAVEICECEK_ORAN', 
'LOKANTAOTEL_ORAN',
'SCHWEPPES',
'DAMLA_WATER', 
'CAPPY', 
'BURN',
'MONSTER',
'ISYERI_YOGUNLUGU_SAYI_KM2_YERLE',
'ISYERI_YOGUNLUGU_SAYI_KM2',
'KONUT_YOGUNLUGU_SAYI_KM2_YERLES',
'KONUT_YOGUNLUGU_SAYI_KM2'
]]


# In[14]:


model_data


# In[15]:


model_data.info()


# In[16]:


# Get one label encoding of column "outlet_ses"
model_data["outlet_ses"] = model_data["outlet_ses"].replace({'A': 6, 'B': 5, 'C1': 4, 'C2': 3, 'D': 2, 'E': 1})


# In[17]:


# Get one label encoding of column "age_cluster"
model_data["age_cluster"] = model_data["age_cluster"].replace({'SIFIR': 1, 'TEEN': 2, 'YOUNG': 3, 'YOUNG ADULT': 4, 'ADULT': 5, 'MIDDLE AGED': 6, 'OLD': 7})


# In[18]:


# Get one label encoding of column "SEGMENT"
model_data["SEGMENT"] = model_data["SEGMENT"].replace({'BRONZE': 1, 'SILVER': 2, 'SILVER PLUS': 3, 'GOLD': 4})


# In[19]:


model_data.info()


# #### Scaling and Normal Dist.

# In[20]:


model_df = model_data.copy()


# In[21]:


model_data.fillna(0,inplace=True)


# In[22]:


model_data.clip(lower=0, inplace=True)


# In[23]:


model_data.replace({0: 0.00001}, inplace=True)


# In[24]:


# define a method to scale data, looping thru the columns, and passing a scaler
def scale_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in data.columns:
        data[col] = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# In[25]:


def normal_dist(data):
    for col in data.columns:
        data[col] = data[col].apply(lambda x: boxcox1p(x,0.25))
        stats.boxcox(data[col])[0]
    return data


# In[26]:


model_data.info()


# In[27]:


# Normal Dist.
model_df["SEGMENT"] = stats.boxcox(model_data["SEGMENT"])[0]
model_df["outlet_ses"] = stats.boxcox(model_data["outlet_ses"])[0]
model_df["age_cluster"] = stats.boxcox(model_data["age_cluster"])[0]      
model_df["YAYA_TRAFIGI"] = stats.boxcox(model_data["YAYA_TRAFIGI"])[0]   
model_df["AYLIK_HARCAMA"] = stats.boxcox(model_data["AYLIK_HARCAMA"])[0]   
model_df["GIDAVEICECEK"] = stats.boxcox(model_data["GIDAVEICECEK"])[0]  
model_df["HANE_BUYUKLUGU"] = stats.boxcox(model_data["HANE_BUYUKLUGU"])[0]
model_df["ALKOLTUTUN"] = stats.boxcox(model_data["ALKOLTUTUN"])[0]   
model_df["EGLENCEKULTUR"] = stats.boxcox(model_data["EGLENCEKULTUR"])[0]  
model_df["LOKANTAOTEL"] = stats.boxcox(model_data["LOKANTAOTEL"])[0]    
model_df["ALKOLTUTUN_ORAN"] = stats.boxcox(model_data["ALKOLTUTUN_ORAN"])[0]     
model_df["EGLENCEKULTUR_ORAN"] = stats.boxcox(model_data["EGLENCEKULTUR_ORAN"])[0] 
model_df["GIDAVEICECEK_ORAN"] = stats.boxcox(model_data["GIDAVEICECEK_ORAN"])[0]  
model_df["LOKANTAOTEL_ORAN"] = stats.boxcox(model_data["LOKANTAOTEL_ORAN"])[0]    
model_df["SCHWEPPES"] = stats.boxcox(model_data["SCHWEPPES"])[0]    
model_df["DAMLA_WATER"] = stats.boxcox(model_data["DAMLA_WATER"])[0]    
model_df["CAPPY"] = stats.boxcox(model_data["CAPPY"])[0]    
model_df["BURN"] = stats.boxcox(model_data["BURN"])[0]    
model_df["MONSTER"] = stats.boxcox(model_data["MONSTER"])[0]   
model_df["ISYERI_YOGUNLUGU_SAYI_KM2_YERLE"] = stats.boxcox(model_data["ISYERI_YOGUNLUGU_SAYI_KM2_YERLE"])[0]   
model_df["ISYERI_YOGUNLUGU_SAYI_KM2"] = stats.boxcox(model_data["ISYERI_YOGUNLUGU_SAYI_KM2"])[0]   
model_df["KONUT_YOGUNLUGU_SAYI_KM2_YERLES"] = stats.boxcox(model_data["KONUT_YOGUNLUGU_SAYI_KM2_YERLES"])[0]   
model_df["KONUT_YOGUNLUGU_SAYI_KM2"] = stats.boxcox(model_data["KONUT_YOGUNLUGU_SAYI_KM2"])[0]   
#model_df["IL_BAZLI_TURIST_YERLI_ISLETME"] = stats.boxcox(model_data["IL_BAZLI_TURIST_YERLI_ISLETME"])[0]   
#model_df["IL_BAZLI_TURIST_YBN_ISLETME"] = stats.boxcox(model_data["IL_BAZLI_TURIST_YBN_ISLETME"])[0]   


# In[28]:


# original value
sns.displot(model_data["AYLIK_HARCAMA"])


# In[29]:


# original to normal dist. 
sns.displot(model_df["AYLIK_HARCAMA"])


# In[30]:


# normal dist. to scaling
sns.displot(model_data["SCHWEPPES"])


# In[31]:


# normal dist. to scaling
sns.displot(model_df["SCHWEPPES"])


# #### Scaling Part

# In[32]:


model_df = scale_data(model_df)


# In[33]:


sns.displot(model_df['outlet_ses'])


# In[34]:


sns.displot(model_df['age_cluster'])


# In[35]:


# final data for model
model_df.info()


# ### Feature Weighting

# In[36]:


# Modele koymadan önce feature'ları ağırlıklandıralım.      
model_df["outlet_ses"] = model_df["outlet_ses"] *  1.5     
model_df["age_cluster"] = model_df["age_cluster"] *  1.5
#model_df["YAYA_TRAFIGI"] = model_df["YAYA_TRAFIGI"] * 2
#model_df["GIDAVEICECEK"] = model_df["GIDAVEICECEK"] * 2
#model_df["LOKANTAOTEL_ORAN"] = model_df["LOKANTAOTEL_ORAN"] * 2
#model_df["GIDAVEICECEK_ORAN"] = model_df["GIDAVEICECEK_ORAN"] * 2   
#model_df["AYLIK_HARCAMA"] = model_df["AYLIK_HARCAMA"] * 2  
#model_df["HANE_BUYUKLUGU"] = model_df["HANE_BUYUKLUGU"] * 2
#model_df["ALKOLTUTUN_ORAN"] = model_df["ALKOLTUTUN_ORAN"] * 30    
#model_df["EGLENCEKULTUR_ORAN"] = model_df["EGLENCEKULTUR_ORAN"] * 0.5 
#model_df["KULTUREL"] = model_df["KULTUREL"] * 10
#model_df["UNIVERSITE"] = model_df["UNIVERSITE"] * 10
#model_df["EGITIM"] = model_df["EGITIM"] * 10 

model_df["GIDAVEICECEK"] = model_df["GIDAVEICECEK"] * 0.5                  
model_df["HANE_BUYUKLUGU"] = model_df["HANE_BUYUKLUGU"] * 0.5              
model_df["ALKOLTUTUN"] = model_df["ALKOLTUTUN"] * 0.5                  
model_df["EGLENCEKULTUR"] = model_df["EGLENCEKULTUR"] * 0.5               
model_df["LOKANTAOTEL"] = model_df["LOKANTAOTEL"] * 0.5                 
model_df["ALKOLTUTUN_ORAN"] = model_df["ALKOLTUTUN_ORAN"] * 0.5             
model_df["EGLENCEKULTUR_ORAN"] = model_df["EGLENCEKULTUR_ORAN"] * 0.5          
model_df["GIDAVEICECEK_ORAN"] = model_df["GIDAVEICECEK_ORAN"] * 0.5           
model_df["LOKANTAOTEL_ORAN"] = model_df["LOKANTAOTEL_ORAN"] * 0.5


# In[37]:


# del model_df["MONSTER"]


# In[38]:


# del model_df["EGLENCEKULTUR_ORAN"]


# ## Model

# In[39]:


hrc_model = AgglomerativeClustering(n_clusters=5, linkage='ward', affinity='euclidean')
hrc_preds = hrc_model.fit_predict(model_df)


# In[40]:


model_df["hrc_cluster"] = hrc_preds


# In[41]:


df["hrc_cluster"] = hrc_preds


# In[42]:


model_df["hrc_cluster"].unique()


# In[43]:


model_df['hrc_cluster'].value_counts().plot(kind='pie')


# In[44]:


model_df['hrc_cluster'].value_counts()


# ## Decision Points

# In[45]:


from sklearn.model_selection import train_test_split

x = model_df.drop('hrc_cluster', axis=1)
y = model_df['hrc_cluster']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
dc = DecisionTreeClassifier(criterion="entropy", random_state=42, ) #max feat. bak bakalım
dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)

d_text = tree.export_text(dc)
print(d_text)


# In[46]:


model_df.info()


# In[47]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# In[48]:


dc.feature_importances_


# In[49]:


pd.Series(dc.feature_importances_, index=x_train.columns).nlargest(35).plot(kind='barh',figsize=(25, 6))


# In[50]:


plt.figure(figsize=(15, 9))
sns.scatterplot(x="COCACOLA", y="SPRITE", hue=df['hrc_cluster'], data=df, palette=['green','red','dodgerblue',"orange","purple"])
plt.title('Hier. Clustering with 2 dimensions')


# In[51]:


#Plot the clusters obtained using k means
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['AYLIK_HARCAMA'],df['GECE_NUFUS'], df['GUNDUZ_NUFUS'],
                     c=hrc_preds, s=20, cmap="rainbow")


ax.set_title('Cluster Dist.')
ax.set_xlabel('AYLIK_HARCAMA')
ax.set_ylabel('GECE_NUFUS')
ax.set_zlabel('GUNDUZ_NUFUS')
plt.show()


# In[52]:


fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['outlet_ses'].replace({'A': 6, 'B': 5, 'C1': 4, 'C2': 3, 'D': 2, 'E': 1}), df['GECE_NUFUS'], df['GUNDUZ_NUFUS'],
                     c=hrc_preds, s=20, cmap="rainbow")


ax.set_title('Cluster Dist.')
ax.set_xlabel('outlet_ses')
ax.set_ylabel('GECE_NUFUS')
ax.set_zlabel('GUNDUZ_NUFUS')
plt.show()


# ## Results

# ----

# In[205]:


df.MONSTER.max()


# In[207]:


df.MONSTER.mean()


# In[204]:


cluster_0.MONSTER.max()


# In[209]:


cluster_2.MONSTER.max()


# In[208]:


cluster_3.MONSTER.max()


# In[198]:


cluster_2.BURN.mean()


# In[199]:


cluster_3.BURN.mean()


# ----

# In[53]:


# extract coordinates
df["x"] = df["GEOGPOINT"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[1])
df["y"] = df["GEOGPOINT"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[0])


# In[54]:


df["y"] = pd.to_numeric(df["y"], downcast="float")
df["x"] = pd.to_numeric(df["x"], downcast="float")


# In[55]:


cluster_0 = df.query("hrc_cluster == 0")
cluster_1 = df.query("hrc_cluster == 1")
cluster_2 = df.query("hrc_cluster == 2")
cluster_3 = df.query("hrc_cluster == 3")
cluster_4 = df.query("hrc_cluster == 4")

https://www.python-graph-gallery.com/radar-chart/
# ### General Info.

# In[56]:


print("Toplam outlet sayısı : ", df.outlet_number.nunique(), "\n")

print("Totaldeki ortalama aylık harcama: ",df['AYLIK_HARCAMA'].mean(), "\n")

print("Ortalama Alkol/Tütün harcaması: ", (df['AYLIK_HARCAMA'].mean() / df.ALKOLTUTUN_ORAN.mean()), "\n")

print("Totaldeki A,B,C1 kategorisindeki toplam outlet sayısı: ", df.query("outlet_ses == 'A' or outlet_ses == 'B' or outlet_ses == 'C1'").outlet_number.nunique(), "\n")

print("'Genç' kategorisi kapsamındaki toplam outlet sayısı: ", df.query("age_cluster == 'YOUNG' or age_cluster == 'TEEN' or age_cluster == 'YOUNG ADULT'").outlet_number.nunique(), "\n")

print("Geneldeki ortalama Alkol/Tütün oranı: ", '%', (df.ALKOLTUTUN_ORAN.mean()), "\n")

print("Ortalama Gıda/İçecek oranı: ", '%', (df.GIDAVEICECEK_ORAN.mean()), "\n")

print("Ortalama Lokanta/Otel oranı: ", '%', (df.LOKANTAOTEL_ORAN.mean()), "\n")

print("Ortalama Eğlence/Kültür oranı: ", '%', (df.EGLENCEKULTUR_ORAN.mean()), "\n")

print("Ortalama hane büyüklüğü: ", df.HANE_BUYUKLUGU.mean(), "\n")

print("Ortalama Eğitim birimi sayısı: ", df.EGITIM.mean(), "\n")

print("Ortalama Üniversite birimi sayısı: ", df.UNIVERSITE.mean(), "\n")


# &nbsp;

# ### Cluster-0 

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama ortalama üstünde.</dd> 
# <dd>2. SES'te C1 sonra C2 en yüksek.</dd>
# <dd>3. Middle age ve Adult çoğunlukta.</dd>
# <dd>4. Alkol/Tütün kullanımı ortalamanın üstünde.</dd>
# </font>

# In[57]:


print("Cluster-0'daki toplam outlet sayısı : ",cluster_0.outlet_number.nunique(), "\n")

print("Cluster-0'ın ortalama aylık harcaması: ",cluster_0['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-0'ın ortalama Alkol/Tütün harcaması: ", (cluster_0['AYLIK_HARCAMA'].mean() / cluster_0.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-0'ın A,B,C1 kategorisindeki outlet oranı: ", '%' ,len(cluster_0.query("outlet_ses == 'A' or outlet_ses == 'B' or outlet_ses == 'C1'").outlet_number)/len(cluster_0.outlet_number)*100, "\n")

print("Cluster-0'ın 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_0.query("age_cluster == 'YOUNG' or age_cluster == 'TEEN' or age_cluster == 'YOUNG ADULT'").outlet_number)/len(cluster_0.outlet_number)*100, "\n")

print("Cluster-0'ın 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_0.query("age_cluster == 'OLD'").outlet_number)/len(cluster_0.outlet_number)*100, "\n")

print("Cluster-0'ın clusterlar genelindeki Alkol/Tütün oranı: ", '%', (cluster_0.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")

print("Cluster-1'in kendi içindeki ortalama Alkol/Tütün oranı: ", '%', (cluster_1.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-0'ın ortalama Gıda/İçecek oranı: ", '%', (cluster_0.GIDAVEICECEK_ORAN.mean()), "\n")

print("Cluster-0'ın ortalama Lokanta/Otel oranı: ", '%', (cluster_0.LOKANTAOTEL_ORAN.mean()), "\n")

print("Cluster-0'ın ortalama Eğlence/Kültür oranı: ", '%', (cluster_0.EGLENCEKULTUR_ORAN.mean()), "\n")

print("Cluster-0'ın ortalama hane büyüklüğü: ", cluster_0.HANE_BUYUKLUGU.mean(), "\n")

print("Cluster-0'ın ortalama Eğitim birimi sayısı: ", cluster_0.EGITIM.mean(), "\n")

print("Cluster-0'ın ortalama Üniversite birimi sayısı: ", cluster_0.UNIVERSITE.mean(), "\n")


# &nbsp;

# ### Cluster-1 

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama ortalama altında.</dd> 
# <dd>2. SES'te C1 ve C2 en yüksek.</dd>
# <dd>3. Young ve Adult en fazla.</dd>
# <dd>4. Alkol/Tütün kullanımı ortalama üstü.</dd>
# </font>

# In[58]:


print("Cluster-1'deki toplam outlet sayısı : ",cluster_1.outlet_number.nunique(), "\n")

print("Cluster-1'in ortalama aylık harcaması: ",cluster_1['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-1'in ortalama Alkol/Tütün harcaması: ", (cluster_1['AYLIK_HARCAMA'].mean() / cluster_1.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-1'in A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_1.query("outlet_ses == 'A' or outlet_ses == 'B' or outlet_ses == 'C1'").outlet_number)/len(cluster_1.outlet_number)*100, "\n")

print("Cluster-1'in 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_1.query("age_cluster == 'YOUNG' or age_cluster == 'TEEN' or age_cluster == 'YOUNG ADULT'").outlet_number)/len(cluster_1.outlet_number)*100, "\n")

print("Cluster-1'in 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_1.query("age_cluster == 'OLD'").outlet_number)/len(cluster_1.outlet_number)*100, "\n")

print("Cluster-1'in clusterlar genelindeki Alkol/Tütün oranı: ", '%', (cluster_1.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")

print("Cluster-1'in kendi içindeki ortalama Alkol/Tütün oranı: ", '%', (cluster_1.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-1'in ortalama Gıda/İçecek oranı: ", '%', (cluster_1.GIDAVEICECEK_ORAN.mean()), "\n")

print("Cluster-1'in ortalama Lokanta/Otel oranı: ", '%', (cluster_1.LOKANTAOTEL_ORAN.mean()), "\n")

print("Cluster-1'in ortalama Eğlence/Kültür oranı: ", '%', (cluster_1.EGLENCEKULTUR_ORAN.mean()), "\n")

print("Cluster-1'in ortalama hane büyüklüğü: ", cluster_1.HANE_BUYUKLUGU.mean(), "\n")

print("Cluster-1'in ortalama Eğitim birimi sayısı: ", cluster_1.EGITIM.mean(), "\n")

print("Cluster-1'in ortalama Üniversite birimi sayısı: ", cluster_1.UNIVERSITE.mean(), "\n")


# &nbsp;

# ### Cluster-2 

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama yüksek.</dd> 
# <dd>2. SES'te C1 en fazla.</dd>
# <dd>3. Middle age ve Old fazla en fazla.</dd>
# <dd>4. Alkol/Tütün kullanımı ortalama.</dd>
# </font>

# In[242]:


print("Cluster-2'deki toplam outlet sayısı : ",cluster_2.outlet_number.nunique(), "\n")

print("Cluster-2'nin ortalama aylık harcaması: ",cluster_2['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-2'nin ortalama Alkol/Tütün harcaması: ", (cluster_2['AYLIK_HARCAMA'].mean() / cluster_2.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-2'nin A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_2.query("outlet_ses == 'A' or outlet_ses == 'B' or outlet_ses == 'C1'").outlet_number)/len(cluster_2.outlet_number)*100, "\n")

print("Cluster-2'nin 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_2.query("age_cluster == 'YOUNG' or age_cluster == 'TEEN' or age_cluster == 'YOUNG ADULT'").outlet_number)/len(cluster_2.outlet_number)*100, "\n")

print("Cluster-2'nin 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_2.query("age_cluster == 'OLD'").outlet_number)/len(cluster_2.outlet_number)*100, "\n")

print("Cluster-2'nin clusterlar genelindeki Alkol/Tütün oranı: ", '%', (cluster_2.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")

print("Cluster-2'in kendi içindeki ortalama Alkol/Tütün oranı: ", '%', (cluster_2.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-2'in ortalama Gıda/İçecek oranı: ", '%', (cluster_2.GIDAVEICECEK_ORAN.mean()), "\n")

print("Cluster-2'in ortalama Lokanta/Otel oranı: ", '%', (cluster_2.LOKANTAOTEL_ORAN.mean()), "\n")

print("Cluster-2'in ortalama Eğlence/Kültür oranı: ", '%', (cluster_2.EGLENCEKULTUR_ORAN.mean()), "\n")

print("Cluster-2'in ortalama hane büyüklüğü: ", cluster_2.HANE_BUYUKLUGU.mean(), "\n")


# &nbsp;

# ### Cluster-3 

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama ortalama altında.</dd> 
# <dd>2. SES'te C2 ve C1 yoğunlukta.</dd>
# <dd>3. Young ve Young Adult fazla. </dd>
# <dd>4. Alkol/Tütün kullanımı yüksek.</dd>
# </font>

# In[63]:


print("Cluster-3'deki toplam outlet sayısı : ",cluster_3.outlet_number.nunique(), "\n")

print("Cluster-3'ün ortalama aylık harcaması: ",cluster_3['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-3'in ortalama Alkol/Tütün harcaması: ", (cluster_3['AYLIK_HARCAMA'].mean() / cluster_3.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-3'ün A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_3.query("outlet_ses == 'A' or outlet_ses == 'B' or outlet_ses == 'C1'").outlet_number)/len(cluster_3.outlet_number)*100, "\n")

print("Cluster-3'ün 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_3.query("age_cluster == 'YOUNG' or age_cluster == 'TEEN' or age_cluster == 'YOUNG ADULT'").outlet_number)/len(cluster_3.outlet_number)*100, "\n")

print("Cluster-3'ün 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_3.query("age_cluster == 'OLD'").outlet_number)/len(cluster_3.outlet_number)*100, "\n")

print("Cluster-3'ün clusterlar genelindeki Alkol/Tütün oranı: ", '%', (cluster_3.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")

print("Cluster-3'ün kendi içindeki ortalama Alkol/Tütün oranı: ", '%', (cluster_3.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-3'ün ortalama Gıda/İçecek oranı: ", '%', (cluster_3.GIDAVEICECEK_ORAN.mean()), "\n")

print("Cluster-3'ün ortalama Lokanta/Otel oranı: ", '%', (cluster_3.LOKANTAOTEL_ORAN.mean()), "\n")

print("Cluster-3'ün ortalama Eğlence/Kültür oranı: ", '%', (cluster_3.EGLENCEKULTUR_ORAN.mean()), "\n")

print("Cluster-3'ün ortalama hane büyüklüğü: ", cluster_3.HANE_BUYUKLUGU.mean(), "\n")


# &nbsp;

# ### Cluster-4 

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama en yüksek.</dd> 
# <dd>2. SES'te B yoğunlukta.</dd>
# <dd>3. Old sınıfı en yoğunlukta.</dd>
# <dd>4. Alkol/Tütün kullanımı düşük.</dd>
# </font>

# In[61]:


print("Cluster-4'deki toplam outlet sayısı : ",cluster_4.outlet_number.nunique(), "\n")

print("Cluster-4'ün ortalama aylık harcaması: ",cluster_4['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-4'ün ortalama Alkol/Tütün harcaması: ", (cluster_4['AYLIK_HARCAMA'].mean() / cluster_4.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-4'ün A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_4.query("outlet_ses == 'A' or outlet_ses == 'B' or outlet_ses == 'C1'").outlet_number)/len(cluster_4.outlet_number)*100, "\n")

print("Cluster-4'ün 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_4.query("age_cluster == 'YOUNG' or age_cluster == 'TEEN' or age_cluster == 'YOUNG ADULT'").outlet_number)/len(cluster_4.outlet_number)*100, "\n")

print("Cluster-4'ün 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_4.query("age_cluster == 'OLD'").outlet_number)/len(cluster_4.outlet_number)*100, "\n")

print("Cluster-4'ün clusterlar genelindeki Alkol/Tütün oranı: ", '%', (cluster_4.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")

print("Cluster-4'ün kendi içindeki ortalama Alkol/Tütün oranı: ", '%', (cluster_4.ALKOLTUTUN_ORAN.mean()), "\n")

print("Cluster-4'ün ortalama Gıda/İçecek oranı: ", '%', (cluster_4.GIDAVEICECEK_ORAN.mean()), "\n")

print("Cluster-4'ün ortalama Lokanta/Otel oranı: ", '%', (cluster_4.LOKANTAOTEL_ORAN.mean()), "\n")

print("Cluster-4'ün ortalama Eğlence/Kültür oranı: ", '%', (cluster_4.EGLENCEKULTUR_ORAN.mean()), "\n")

print("Cluster-4'ün ortalama hane büyüklüğü: ", cluster_4.HANE_BUYUKLUGU.mean(), "\n")


# In[62]:


breakhere


# <h1><center>Comparison</center></h1>

# ![](age.png "Age Dist.")

# ![](info.png "SES Dist.")

# ![](ses.png "SES Dist.")

# <h1><center>Cluster-0</center></h1>

# In[102]:


pie=cluster_0.groupby('age_cluster').size().reset_index()
pie.columns=['age_cluster','value']
px.pie(pie,values='value',names='age_cluster', title='Age Dist. in CLUSTER-0')


# In[103]:


pie=cluster_0.groupby('outlet_ses').size().reset_index()
pie.columns=['outlet_ses','value']
px.pie(pie,values='value',names='outlet_ses', title='SES Dist. in CLUSTER-0')


# In[104]:


yas_brand_0 = cluster_0[['age_cluster'
,'BURN'                
,'CAPPY'               
,'CC_LIGHT'                   
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].groupby(['age_cluster']).sum().reset_index()


# In[105]:


yas_brand_0


# In[106]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_0 = yas_brand_0.query("age_cluster=='YOUNG'").groupby("age_cluster").sum().T


# In[107]:


fig = px.pie(yas_cluster_0.reset_index(), values='YOUNG', names="index", title='Brand Dist. in Young Category (CLUSTER-0)')
fig.show()


# In[108]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_teen_0 = yas_brand_0.query("age_cluster=='YOUNG ADULT'").groupby("age_cluster").sum().T


# In[109]:


fig = px.pie(yas_cluster_teen_0.reset_index(), values='YOUNG ADULT', names="index", title='Brand Dist. in Young Adult Category (CLUSTER-0)')
fig.show()


# In[110]:


IC_FC_c0 = pd.DataFrame()

IC_FC_c0['CAPPY_IC_FC'] = cluster_0['CAPPY_IC'] / cluster_0['CAPPY_FC']
IC_FC_c0['CC_LIGHT_IC_FC'] = cluster_0['CC_LIGHT_IC'] / cluster_0['CC_LIGHT_FC']
IC_FC_c0['CC_NO_SUGAR_IC_FC'] = cluster_0['CC_NO_SUGAR_IC'] / cluster_0['CC_NO_SUGAR_FC']
IC_FC_c0['COCACOLA_IC_FC'] = cluster_0['COCACOLA_IC'] / cluster_0['COCACOLA_FC']    
IC_FC_c0['COCACOLA_ENERGY_IC_FC'] = cluster_0['COCACOLA_ENERGY_IC'] / cluster_0['COCACOLA_ENERGY_FC']
IC_FC_c0['DAMLA_MINERA_IC_FC'] = cluster_0['DAMLA_MINERA_IC'] / cluster_0['DAMLA_MINERA_FC']
IC_FC_c0['DAMLA_WATER_IC_FC'] = cluster_0['DAMLA_WATER_IC'] / cluster_0['DAMLA_WATER_FC']
IC_FC_c0['FUSETEA_IC_FC'] = cluster_0['FUSETEA_IC'] / cluster_0['FUSETEA_FC']
IC_FC_c0['BURN_IC_FC'] = cluster_0['BURN_IC'] / cluster_0['BURN_FC']
IC_FC_c0['EXOTIC_IC_FC'] = cluster_0['EXOTIC_IC'] / cluster_0['EXOTIC_FC']
IC_FC_c0['FANTA_IC_FC'] = cluster_0['FANTA_IC'] / cluster_0['FANTA_FC']
IC_FC_c0['SPRITE_IC_FC'] = cluster_0['SPRITE_IC'] / cluster_0['SPRITE_FC']
IC_FC_c0['SCHWEPPES_IC_FC'] = cluster_0['SCHWEPPES_IC'] / cluster_0['SCHWEPPES_FC']
IC_FC_c0['POWERADE_IC_FC'] = cluster_0['POWERADE_IC'] / cluster_0['POWERADE_FC']
IC_FC_c0['MONSTER_IC_FC'] = cluster_0['MONSTER_IC'] / cluster_0['MONSTER_FC']

IC_FC_c0 = IC_FC_c0.reset_index(drop=True)
IC_FC_c0 = IC_FC_c0.replace([np.inf, -np.inf], np.nan)
IC_FC_c0.fillna(0, inplace=True)
IC_FC_c0 = IC_FC_c0.sum().reset_index()
IC_FC_c0.rename({0: "value"}, inplace=True, axis=1)


# In[111]:


fig = px.pie(IC_FC_c0, values='value', names="index", title='IC/FC Dist. in CLUSTER-0')
fig.show()


# In[112]:


c0_rb200 = cluster_0[['burn_rb200',
 'cappy_rb200',
 'cc_light_rb200',
 'cc_rb200',
 'damla_minera_rb200',
 'damla_water_rb200',
 'exotic_rb200',
 'fanta_rb200',
 'fusetea_rb200',
 'monster_rb200',
 'powerade_rb200',
 'schweppes_rb200',
 'sprite_rb200']].reset_index(drop=True)

c0_rb200.fillna(0, inplace=True)
c0_rb200 = c0_rb200.sum().reset_index()
c0_rb200.rename({0: "value"}, inplace=True, axis=1)


# In[113]:


fig = px.pie(c0_rb200, values='value', names="index", title='RB200 Dist. in CLUSTER-0')
fig.show()


# In[114]:


c0_product = cluster_0[['STILL','SPARKLING','WATER']].sum().reset_index()


# In[115]:


c0_product.rename({0: "value"}, inplace=True, axis=1)
c0_product = c0_product.iloc[1: , :]


# In[116]:


fig = px.pie(c0_product, values='value', names="index", title='Sparkling Water and Still Dist. in CLUSTER-0')
fig.show()


# <h1><center>Cluster-1</center></h1>

# In[117]:


pie=cluster_1.groupby('age_cluster').size().reset_index()
pie.columns=['age_cluster','value']
px.pie(pie,values='value',names='age_cluster', title='Age Dist. in CLUSTER-1')


# In[118]:


pie=cluster_1.groupby('outlet_ses').size().reset_index()
pie.columns=['outlet_ses','value']
px.pie(pie,values='value',names='outlet_ses', title='SES Dist. in CLUSTER-1')


# In[119]:


yas_brand_1 = cluster_1[['age_cluster','BURN'                
,'CAPPY'               
,'CC_LIGHT'               
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].groupby(['age_cluster']).sum().reset_index()


# In[120]:


yas_cluster_1 = yas_brand_1.query("age_cluster=='YOUNG'").groupby("age_cluster").sum().T


# In[121]:


fig = px.pie(yas_cluster_1.reset_index(), values='YOUNG', names="index", title='Brand Dist. in Young Category (CLUSTER-1)')
fig.show()


# In[122]:


yas_cluster_adult = yas_brand_1.query("age_cluster=='YOUNG ADULT'").groupby("age_cluster").sum().T


# In[123]:


fig = px.pie(yas_cluster_adult.reset_index(), values='YOUNG ADULT', names="index", title='Brand Dist. in YOUNG ADULT Category (CLUSTER-1)')
fig.show()


# In[124]:


IC_FC_c1 = pd.DataFrame()

IC_FC_c1['CAPPY_IC_FC'] = cluster_1['CAPPY_IC'] / cluster_1['CAPPY_FC']
IC_FC_c1['CC_LIGHT_IC_FC'] = cluster_1['CC_LIGHT_IC'] / cluster_1['CC_LIGHT_FC']
IC_FC_c1['CC_NO_SUGAR_IC_FC'] = cluster_1['CC_NO_SUGAR_IC'] / cluster_1['CC_NO_SUGAR_FC']
IC_FC_c1['COCACOLA_IC_FC'] = cluster_1['COCACOLA_IC'] / cluster_1['COCACOLA_FC']    
IC_FC_c1['COCACOLA_ENERGY_IC_FC'] = cluster_1['COCACOLA_ENERGY_IC'] / cluster_1['COCACOLA_ENERGY_FC']
IC_FC_c1['DAMLA_MINERA_IC_FC'] = cluster_1['DAMLA_MINERA_IC'] / cluster_1['DAMLA_MINERA_FC']
IC_FC_c1['DAMLA_WATER_IC_FC'] = cluster_1['DAMLA_WATER_IC'] / cluster_1['DAMLA_WATER_FC']
IC_FC_c1['FUSETEA_IC_FC'] = cluster_1['FUSETEA_IC'] / cluster_1['FUSETEA_FC']
IC_FC_c1['BURN_IC_FC'] = cluster_1['BURN_IC'] / cluster_1['BURN_FC']
IC_FC_c1['EXOTIC_IC_FC'] = cluster_1['EXOTIC_IC'] / cluster_1['EXOTIC_FC']
IC_FC_c1['FANTA_IC_FC'] = cluster_1['FANTA_IC'] / cluster_1['FANTA_FC']
IC_FC_c1['SPRITE_IC_FC'] = cluster_1['SPRITE_IC'] / cluster_1['SPRITE_FC']
IC_FC_c1['SCHWEPPES_IC_FC'] = cluster_1['SCHWEPPES_IC'] / cluster_1['SCHWEPPES_FC']
IC_FC_c1['POWERADE_IC_FC'] = cluster_1['POWERADE_IC'] / cluster_1['POWERADE_FC']
IC_FC_c1['MONSTER_IC_FC'] = cluster_1['MONSTER_IC'] / cluster_1['MONSTER_FC']

IC_FC_c1 = IC_FC_c1.reset_index(drop=True)
IC_FC_c1 = IC_FC_c1.replace([np.inf, -np.inf], np.nan)
IC_FC_c1.fillna(0, inplace=True)
IC_FC_c1 = IC_FC_c1.sum().reset_index()
IC_FC_c1.rename({0: "value"}, inplace=True, axis=1)


# In[125]:


fig = px.pie(IC_FC_c1, values='value', names="index", title='IC/FC Dist. in CLUSTER-1')
fig.show()


# In[126]:


c1_rb200 = cluster_1[['burn_rb200',
 'cappy_rb200',
 'cc_light_rb200',
 'cc_rb200',
 'damla_minera_rb200',
 'damla_water_rb200',
 'exotic_rb200',
 'fanta_rb200',
 'fusetea_rb200',
 'monster_rb200',
 'powerade_rb200',
 'schweppes_rb200',
 'sprite_rb200']].reset_index(drop=True)

c1_rb200.fillna(0, inplace=True)
c1_rb200 = c1_rb200.sum().reset_index()
c1_rb200.rename({0: "value"}, inplace=True, axis=1)


# In[127]:


fig = px.pie(c1_rb200, values='value', names="index", title='RB200 Dist. in CLUSTER-1')
fig.show()


# In[128]:


c1_product = cluster_1[['STILL','SPARKLING','WATER']].sum().reset_index()


# In[129]:


c1_product.rename({0: "value"}, inplace=True, axis=1)
c1_product = c1_product.iloc[1: , :]


# In[130]:


fig = px.pie(c1_product, values='value', names="index", title='Sparkling Water and Still Dist. in CLUSTER-1')
fig.show()


# <h1><center>Cluster-2</center></h1>
# 
# 

# In[131]:


pie=cluster_2.groupby('age_cluster').size().reset_index()
pie.columns=['age_cluster','value']
px.pie(pie,values='value',names='age_cluster', title='Age Dist. in CLUSTER-2')


# In[132]:


pie=cluster_2.groupby('outlet_ses').size().reset_index()
pie.columns=['outlet_ses','value']
px.pie(pie,values='value',names='outlet_ses', title='SES Dist. in CLUSTER-2')


# In[133]:


yas_brand_2 = cluster_2[['age_cluster','BURN'                
,'CAPPY'               
,'CC_LIGHT'                    
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].groupby(['age_cluster']).sum().reset_index()


# In[134]:


yas_brand_2


# In[135]:


yas_cluster_2 = yas_brand_2.query("age_cluster=='MIDDLE AGED'").groupby("age_cluster").sum().T


# In[136]:


yas_cluster_2


# In[137]:


yas_brand_2_sum = cluster_2[['age_cluster','BURN'                
,'CAPPY'               
,'CC_LIGHT'                    
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].sum().reset_index()


# In[138]:


yas_brand_2_sum.rename({0: "value"}, inplace=True, axis=1)
yas_brand_2_sum = yas_brand_2_sum.iloc[1: , :]


# In[139]:


yas_brand_2_sum


# In[140]:


fig = px.pie(yas_brand_2_sum, values='value', names="index", title='Brand Dist. in CLUSTER-2')
fig.show()


# In[141]:


fig = px.pie(yas_cluster_2.reset_index(), values='MIDDLE AGED', names="index", title='Brand Dist. in Middle Aged Category (CLUSTER-2)')
fig.show()


# In[146]:


yas_cluster_2_old = yas_brand_2.query("age_cluster=='YOUNG'").groupby("age_cluster").sum().T


# In[147]:


fig = px.pie(yas_cluster_2_old.reset_index(), values='YOUNG', names="index", title='Brand Dist. in YOUNG Category (CLUSTER-2)')
fig.show()


# In[149]:


IC_FC_c2 = pd.DataFrame()

IC_FC_c2['CAPPY_IC_FC'] = cluster_2['CAPPY_IC'] / cluster_2['CAPPY_FC']
IC_FC_c2['CC_LIGHT_IC_FC'] = cluster_2['CC_LIGHT_IC'] / cluster_2['CC_LIGHT_FC']
IC_FC_c2['CC_NO_SUGAR_IC_FC'] = cluster_2['CC_NO_SUGAR_IC'] / cluster_2['CC_NO_SUGAR_FC']
IC_FC_c2['COCACOLA_IC_FC'] = cluster_2['COCACOLA_IC'] / cluster_2['COCACOLA_FC']    
IC_FC_c2['COCACOLA_ENERGY_IC_FC'] = cluster_2['COCACOLA_ENERGY_IC'] / cluster_2['COCACOLA_ENERGY_FC']
IC_FC_c2['DAMLA_MINERA_IC_FC'] = cluster_2['DAMLA_MINERA_IC'] / cluster_2['DAMLA_MINERA_FC']
IC_FC_c2['DAMLA_WATER_IC_FC'] = cluster_2['DAMLA_WATER_IC'] / cluster_2['DAMLA_WATER_FC']
IC_FC_c2['FUSETEA_IC_FC'] = cluster_2['FUSETEA_IC'] / cluster_2['FUSETEA_FC']
IC_FC_c2['BURN_IC_FC'] = cluster_2['BURN_IC'] / cluster_2['BURN_FC']
IC_FC_c2['EXOTIC_IC_FC'] = cluster_2['EXOTIC_IC'] / cluster_2['EXOTIC_FC']
IC_FC_c2['FANTA_IC_FC'] = cluster_2['FANTA_IC'] / cluster_2['FANTA_FC']
IC_FC_c2['SPRITE_IC_FC'] = cluster_2['SPRITE_IC'] / cluster_2['SPRITE_FC']
IC_FC_c2['SCHWEPPES_IC_FC'] = cluster_2['SCHWEPPES_IC'] / cluster_2['SCHWEPPES_FC']
IC_FC_c2['POWERADE_IC_FC'] = cluster_2['POWERADE_IC'] / cluster_2['POWERADE_FC']
IC_FC_c2['MONSTER_IC_FC'] = cluster_2['MONSTER_IC'] / cluster_2['MONSTER_FC']

IC_FC_c2 = IC_FC_c2.reset_index(drop=True)
IC_FC_c2 = IC_FC_c2.replace([np.inf, -np.inf], np.nan)
IC_FC_c2.fillna(0, inplace=True)
IC_FC_c2 = IC_FC_c2.sum().reset_index()
IC_FC_c2.rename({0: "value"}, inplace=True, axis=1)


# In[150]:


fig = px.pie(IC_FC_c2, values='value', names="index", title='IC/FC Dist. in CLUSTER-2')
fig.show()


# In[151]:


c2_rb200 = cluster_2[['burn_rb200',
 'cappy_rb200',
 'cc_light_rb200',
 'cc_rb200',
 'damla_minera_rb200',
 'damla_water_rb200',
 'exotic_rb200',
 'fanta_rb200',
 'fusetea_rb200',
 'monster_rb200',
 'powerade_rb200',
 'schweppes_rb200',
 'sprite_rb200']].reset_index(drop=True)

c2_rb200.fillna(0, inplace=True)
c2_rb200 = c2_rb200.sum().reset_index()
c2_rb200.rename({0: "value"}, inplace=True, axis=1)


# In[152]:


fig = px.pie(c2_rb200, values='value', names="index", title='RB200 Dist. in CLUSTER-2')
fig.show()


# In[153]:


c2_product = cluster_2[['STILL','SPARKLING','WATER']].sum().reset_index()


# In[154]:


c2_product.rename({0: "value"}, inplace=True, axis=1)
c2_product = c2_product.iloc[1: , :]


# In[155]:


fig = px.pie(c2_product, values='value', names="index", title='Sparkling Water and Still Dist. in CLUSTER-2')
fig.show()


# <h1><center>Cluster-3</center></h1>

# In[156]:


pie=cluster_3.groupby('age_cluster').size().reset_index()
pie.columns=['age_cluster','value']
px.pie(pie,values='value',names='age_cluster', title='Age Dist. in CLUSTER-3')


# In[157]:


pie=cluster_3.groupby('outlet_ses').size().reset_index()
pie.columns=['outlet_ses','value']
px.pie(pie,values='value',names='outlet_ses', title='SES Dist. in CLUSTER-3')


# In[158]:


yas_brand_3 = cluster_3[['age_cluster','BURN'                
,'CAPPY'               
,'CC_LIGHT'                 
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].groupby(['age_cluster']).sum().reset_index()


# In[159]:


yas_brand_3


# In[160]:


yas_cluster_3 = yas_brand_3.query("age_cluster=='OLD'").groupby("age_cluster").sum().T


# In[161]:


yas_cluster_3


# In[162]:


yas_brand_3_sum = cluster_3[['age_cluster','BURN'                
,'CAPPY'               
,'CC_LIGHT'                     
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].sum().reset_index()


# In[163]:


yas_brand_3_sum.rename({0: "value"}, inplace=True, axis=1)
yas_brand_3_sum = yas_brand_3_sum.iloc[1: , :]


# In[164]:


yas_brand_3_sum


# In[165]:


fig = px.pie(yas_brand_3_sum, values='value', names="index", title='Brand Dist. in CLUSTER-3')
fig.show()


# In[166]:


IC_FC_c3 = pd.DataFrame()

IC_FC_c3['CAPPY_IC_FC'] = cluster_3['CAPPY_IC'] / cluster_3['CAPPY_FC']
IC_FC_c3['CC_LIGHT_IC_FC'] = cluster_3['CC_LIGHT_IC'] / cluster_3['CC_LIGHT_FC']
IC_FC_c3['CC_NO_SUGAR_IC_FC'] = cluster_3['CC_NO_SUGAR_IC'] / cluster_3['CC_NO_SUGAR_FC']
IC_FC_c3['COCACOLA_IC_FC'] = cluster_3['COCACOLA_IC'] / cluster_3['COCACOLA_FC']    
IC_FC_c3['COCACOLA_ENERGY_IC_FC'] = cluster_3['COCACOLA_ENERGY_IC'] / cluster_3['COCACOLA_ENERGY_FC']
IC_FC_c3['DAMLA_MINERA_IC_FC'] = cluster_3['DAMLA_MINERA_IC'] / cluster_3['DAMLA_MINERA_FC']
IC_FC_c3['DAMLA_WATER_IC_FC'] = cluster_3['DAMLA_WATER_IC'] / cluster_3['DAMLA_WATER_FC']
IC_FC_c3['FUSETEA_IC_FC'] = cluster_2['FUSETEA_IC'] / cluster_3['FUSETEA_FC']
IC_FC_c3['BURN_IC_FC'] = cluster_3['BURN_IC'] / cluster_3['BURN_FC']
IC_FC_c3['EXOTIC_IC_FC'] = cluster_3['EXOTIC_IC'] / cluster_3['EXOTIC_FC']
IC_FC_c3['FANTA_IC_FC'] = cluster_3['FANTA_IC'] / cluster_3['FANTA_FC']
IC_FC_c3['SPRITE_IC_FC'] = cluster_3['SPRITE_IC'] / cluster_3['SPRITE_FC']
IC_FC_c3['SCHWEPPES_IC_FC'] = cluster_3['SCHWEPPES_IC'] / cluster_3['SCHWEPPES_FC']
IC_FC_c3['POWERADE_IC_FC'] = cluster_3['POWERADE_IC'] / cluster_3['POWERADE_FC']
IC_FC_c3['MONSTER_IC_FC'] = cluster_3['MONSTER_IC'] / cluster_3['MONSTER_FC']

IC_FC_c3 = IC_FC_c3.reset_index(drop=True)
IC_FC_c3 = IC_FC_c3.replace([np.inf, -np.inf], np.nan)
IC_FC_c3.fillna(0, inplace=True)
IC_FC_c3 = IC_FC_c3.sum().reset_index()
IC_FC_c3.rename({0: "value"}, inplace=True, axis=1)


# In[167]:


fig = px.pie(IC_FC_c3, values='value', names="index", title='IC/FC Dist. in CLUSTER-3')
fig.show()


# In[168]:


c3_rb200 = cluster_3[['burn_rb200',
 'cappy_rb200',
 'cc_light_rb200',
 'cc_rb200',
 'damla_minera_rb200',
 'damla_water_rb200',
 'exotic_rb200',
 'fanta_rb200',
 'fusetea_rb200',
 'monster_rb200',
 'powerade_rb200',
 'schweppes_rb200',
 'sprite_rb200']].reset_index(drop=True)

c3_rb200.fillna(0, inplace=True)
c3_rb200 = c3_rb200.sum().reset_index()
c3_rb200.rename({0: "value"}, inplace=True, axis=1)


# In[169]:


fig = px.pie(c3_rb200, values='value', names="index", title='RB200 Dist. in CLUSTER-3')
fig.show()


# In[170]:


c3_product = cluster_3[['STILL','SPARKLING','WATER']].sum().reset_index()


# In[171]:


c3_product.rename({0: "value"}, inplace=True, axis=1)
c3_product = c3_product.iloc[1: , :]


# In[172]:


fig = px.pie(c3_product, values='value', names="index", title='Sparkling Water and Still Dist. in CLUSTER-3')
fig.show()


# <h1><center>Cluster-4</center></h1>

# In[173]:


pie=cluster_4.groupby('age_cluster').size().reset_index()
pie.columns=['age_cluster','value']
px.pie(pie,values='value',names='age_cluster', title='Age Dist. in CLUSTER-4')


# In[241]:


pie=cluster_4.groupby('outlet_ses').size().reset_index()
pie.columns=['outlet_ses','value']
px.pie(pie,values='value',names='outlet_ses', title='SES Dist. in CLUSTER-4')


# In[175]:


yas_brand_4 = cluster_4[['age_cluster','BURN'                
,'CAPPY'               
,'CC_LIGHT'                 
,'COCACOLA'            
,'COCACOLA_ENERGY'     
,'DAMLA_MINERA'        
,'DAMLA_WATER'         
,'EXOTIC'         
,'FANTA'         
,'FUSETEA'         
,'MONSTER'         
,'POWERADE'         
,'SCHWEPPES'         
,'SPRITE']].groupby(['age_cluster']).sum().reset_index()


# In[176]:


yas_brand_4


# In[177]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_4 = yas_brand_4.query("age_cluster=='OLD'").groupby("age_cluster").sum().T


# In[178]:


fig = px.pie(yas_cluster_4.reset_index(), values='OLD', names="index", title='Brand Dist. in Old Category (CLUSTER-4)')
fig.show()


# In[179]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_mid_4 = yas_brand_4.query("age_cluster=='MIDDLE AGED'").groupby("age_cluster").sum().T


# In[180]:


fig = px.pie(yas_cluster_mid_4.reset_index(), values='MIDDLE AGED', names="index", title='Brand Dist. in Middle Age Category (CLUSTER-4)')
fig.show()


# In[181]:


IC_FC_c4 = pd.DataFrame()

IC_FC_c4['CAPPY_IC_FC'] = cluster_4['CAPPY_IC'] / cluster_4['CAPPY_FC']
IC_FC_c4['CC_LIGHT_IC_FC'] = cluster_4['CC_LIGHT_IC'] / cluster_4['CC_LIGHT_FC']
IC_FC_c4['CC_NO_SUGAR_IC_FC'] = cluster_4['CC_NO_SUGAR_IC'] / cluster_4['CC_NO_SUGAR_FC']
IC_FC_c4['COCACOLA_IC_FC'] = cluster_4['COCACOLA_IC'] / cluster_4['COCACOLA_FC']    
IC_FC_c4['COCACOLA_ENERGY_IC_FC'] = cluster_4['COCACOLA_ENERGY_IC'] / cluster_4['COCACOLA_ENERGY_FC']
IC_FC_c4['DAMLA_MINERA_IC_FC'] = cluster_4['DAMLA_MINERA_IC'] / cluster_4['DAMLA_MINERA_FC']
IC_FC_c4['DAMLA_WATER_IC_FC'] = cluster_4['DAMLA_WATER_IC'] / cluster_4['DAMLA_WATER_FC']
IC_FC_c4['FUSETEA_IC_FC'] = cluster_4['FUSETEA_IC'] / cluster_4['FUSETEA_FC']
IC_FC_c4['BURN_IC_FC'] = cluster_4['BURN_IC'] / cluster_4['BURN_FC']
IC_FC_c4['EXOTIC_IC_FC'] = cluster_4['EXOTIC_IC'] / cluster_4['EXOTIC_FC']
IC_FC_c4['FANTA_IC_FC'] = cluster_4['FANTA_IC'] / cluster_4['FANTA_FC']
IC_FC_c4['SPRITE_IC_FC'] = cluster_4['SPRITE_IC'] / cluster_4['SPRITE_FC']
IC_FC_c4['SCHWEPPES_IC_FC'] = cluster_4['SCHWEPPES_IC'] / cluster_4['SCHWEPPES_FC']
IC_FC_c4['POWERADE_IC_FC'] = cluster_4['POWERADE_IC'] / cluster_4['POWERADE_FC']
IC_FC_c4['MONSTER_IC_FC'] = cluster_4['MONSTER_IC'] / cluster_4['MONSTER_FC']

IC_FC_c4 = IC_FC_c4.reset_index(drop=True)
IC_FC_c4 = IC_FC_c4.replace([np.inf, -np.inf], np.nan)
IC_FC_c4.fillna(0, inplace=True)
IC_FC_c4 = IC_FC_c4.sum().reset_index()
IC_FC_c4.rename({0: "value"}, inplace=True, axis=1)


# In[182]:


fig = px.pie(IC_FC_c4, values='value', names="index", title='IC/FC Dist. in CLUSTER-4')
fig.show()


# In[183]:


IC_c4 = cluster_4[['CAPPY_IC',
'CC_LIGHT_IC',
'CC_NO_SUGAR_IC',
'COCACOLA_IC',    
'COCACOLA_ENERGY_IC',
'DAMLA_MINERA_IC',
'DAMLA_WATER_IC',
'FUSETEA_IC',
'BURN_IC',
'EXOTIC_IC',
'FANTA_IC',
'SPRITE_IC',
'SCHWEPPES_IC',
'POWERADE_IC',
'MONSTER_IC']].sum().reset_index()


# In[184]:


IC_c4.rename({0: "value"}, inplace=True, axis=1)
IC_c4 = IC_c4.iloc[1: , :]


# In[185]:


fig = px.pie(IC_c4, values='value', names="index", title='IC Dist. in CLUSTER-4')
fig.show()


# In[186]:


FC_c4= cluster_4[['CAPPY_FC',
'CC_LIGHT_FC',
'CC_NO_SUGAR_FC',
'COCACOLA_FC',
'COCACOLA_ENERGY_FC',
'DAMLA_MINERA_FC',
'DAMLA_WATER_FC',
'FUSETEA_FC',
'BURN_FC',
'EXOTIC_FC',
'FANTA_FC',
'MONSTER_FC',
'POWERADE_FC',
'SCHWEPPES_FC',
'SPRITE_FC']].sum().reset_index()


# In[187]:


FC_c4.rename({0: "value"}, inplace=True, axis=1)
FC_c4 = FC_c4.iloc[1: , :]


# In[188]:


fig = px.pie(FC_c4, values='value', names="index", title='FC Dist. in CLUSTER-4')
fig.show()


# In[189]:


c4_product = cluster_4[['STILL','SPARKLING','WATER']].sum().reset_index()


# In[190]:


c4_product.rename({0: "value"}, inplace=True, axis=1)
c4_product = c4_product.iloc[1: , :]


# In[191]:


fig = px.pie(c4_product, values='value', names="index", title='Sparkling Water and Still Dist. in CLUSTER-4')
fig.show()


# ----

# In[192]:


stophere


# In[ ]:


df.to_csv("shopper_clusters.csv")


# In[ ]:


# bigquery'e tabloyu ekleme
# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

client = bigquery.Client()

table_id = "coca-cola-data-lake.predictive_order.shopper_profile_v2" 

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("outlet_name", bigquery.enums.SqlTypeNames.STRING),
    ],  write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(df, table_id, job_config=job_config)  
job.result()  

table = client.get_table(table_id)  
print("Loaded {} rows and {} columns to {}".format(table.num_rows, len(table.schema), table_id))


# In[ ]:


df


# In[ ]:




