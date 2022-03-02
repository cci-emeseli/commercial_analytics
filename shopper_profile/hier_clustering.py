#!/usr/bin/env python
# coding: utf-8

# <h1><center>Shopper Profile</center></h1>

# * Bu notebook Traditional Channel'da İstanbuldaki shopper profile'ları bulmak için hier. clustering'in kullanıldığı bir çalışmadır.

# In[1]:


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


df = pd.read_csv('data/shopper_profile(2).csv')


# In[3]:


df.head()


# ## EDA

# In[4]:


df.info() 


# In[5]:


df.MAIN_CHANNEL.unique()


# In[6]:


df.outletnumber.nunique()


# In[7]:


df['outletnumber'] = df['outletnumber'].astype(object)


# In[8]:


df = df.query("MAIN_CHANNEL == 'TRADITIONAL RETAIL'").reset_index(drop=True)


# In[9]:


# Bu dataframe üstünde oynama yapılmamış ana datadır.
df.outletnumber.nunique()


# #### Gece/Gündüz Oranı

# In[10]:


# bu dataframe, model için kullanılacak dataframe'dir
model_data = df.copy()


# In[11]:


# model_data['GECE_NUFUS'].replace({0: 0.00001}, inplace=True)
# model_data['GUNDUZ_NUFUS'].replace({0: 0.00001}, inplace=True)


# In[12]:


# g = model_data[['GUNDUZ_NUFUS','GECE_NUFUS']]


# In[13]:


# model_data["gündüz_gece_oran"] = g["GUNDUZ_NUFUS"]/g["GECE_NUFUS"]


# In[14]:


# model_data.head(3)


# In[15]:


def gece_gündüz(df):
    
    outletnumbers = df.outletnumber.unique()
    outletnumbers_list = outletnumbers.tolist()
    gece_gündüz_list = []
    
    for i in range(len(outletnumbers)):
        
        outlets = df.loc[df["outletnumber"] == outletnumbers[i]]
        
        if (outlets["GUNDUZ_NUFUS"].iloc[0]>=outlets["GECE_NUFUS"].iloc[0]):
            gece_gündüz_list.append("1")
        else:
            gece_gündüz_list.append("-1")
            
    gece_gündüz_df = pd.DataFrame(gece_gündüz_list)       
    result = pd.concat([df, gece_gündüz_df], axis=1)
    
    return(result)


# In[16]:


model_data = model_data[[       
"outletnumber"
,"SES"           
,"YAS_CLUSTER"
,"HANE_BUYUKLUGU"
,"ZENGINLIK_INDEKSI"
,"AYLIK_HARCAMA"   
,"ALKOLTUTUN_ORAN"      
,"EGLENCEKULTUR_ORAN"   
,"LOKANTAOTEL_ORAN" 
,"EGITIM"
,"UNIVERSITE"
,"KULTUREL"          
,"ORTA_UST"
]]


# In[17]:


model_data


# In[18]:


# Get one label encoding of column "SES"
model_data["SES"] = model_data["SES"].replace({'A': 6, 'B': 5, 'C1': 4, 'C2': 3, 'D': 2, 'E': 1})


# In[19]:


# Get one label encoding of column "SES"
model_data["YAS_CLUSTER"] = model_data["YAS_CLUSTER"].replace({'SIFIR': 1, 'TEEN': 2, 'YOUNG': 3, 'YOUNG ADULT': 4, 'ADULT': 5, 'MIDDLE AGED': 6, 'OLD': 7})


# In[20]:


model_data.info()


# #### Scaling and Normal Dist.

# In[21]:


model_df = model_data.copy()


# In[22]:


model_data = model_data.replace(np.nan,0)


# In[23]:


model_data.clip(lower=0, inplace=True)


# In[24]:


model_data.replace({0: 0.00001}, inplace=True)


# In[25]:


# define a method to scale data, looping thru the columns, and passing a scaler
def scale_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in data.columns:
        data[col] = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# In[26]:


def normal_dist(data):
    for col in data.columns:
        data[col] = data[col].apply(lambda x: boxcox1p(x,0.25))
        stats.boxcox(data[col])[0]
    return data


# In[27]:


# Normal Dist.
model_df["SES"] = stats.boxcox(model_data["SES"])[0]
model_df["YAS_CLUSTER"] = stats.boxcox(model_data["YAS_CLUSTER"])[0]
model_df["HANE_BUYUKLUGU"] = stats.boxcox(model_data["HANE_BUYUKLUGU"])[0]      
model_df["ZENGINLIK_INDEKSI"] = stats.boxcox(model_data["ZENGINLIK_INDEKSI"])[0]   
model_df["AYLIK_HARCAMA"] = stats.boxcox(model_data["AYLIK_HARCAMA"])[0]       
#model_df["GIDAVEICECEK_ORAN"] = stats.boxcox(model_data["GIDAVEICECEK_ORAN"])[0]   
model_df["ALKOLTUTUN_ORAN"] = stats.boxcox(model_data["ALKOLTUTUN_ORAN"])[0]     
model_df["EGLENCEKULTUR_ORAN"] = stats.boxcox(model_data["EGLENCEKULTUR_ORAN"])[0]  
model_df["LOKANTAOTEL_ORAN"] = stats.boxcox(model_data["LOKANTAOTEL_ORAN"])[0]    
model_df["EGITIM"] = stats.boxcox(model_data["EGITIM"])[0]              
model_df["UNIVERSITE"] = stats.boxcox(model_data["UNIVERSITE"])[0]          
#model_df["ORTA"] = stats.boxcox(model_data["ORTA"])[0]                
#model_df["ORTA_ALT"] = stats.boxcox(model_data["ORTA_ALT"])[0]            
model_df["ORTA_UST"] = stats.boxcox(model_data["ORTA_UST"])[0]    
model_df["KULTUREL"] = stats.boxcox(model_data["KULTUREL"])[0]   


# In[28]:


# original value
sns.displot(model_data["AYLIK_HARCAMA"])


# In[29]:


# original to normal dist. 
sns.displot(model_df["AYLIK_HARCAMA"])


# In[30]:


# normal dist. to scaling
sns.displot(model_data["YAS_CLUSTER"])


# In[31]:


# normal dist. to scaling
sns.displot(model_df["YAS_CLUSTER"])


# #### Scaling Part

# In[32]:


model_df = scale_data(model_df)


# In[33]:


model_df.YAS_CLUSTER.describe()


# In[34]:


sns.displot(model_df['SES'])


# In[35]:


sns.displot(model_df['YAS_CLUSTER'])


# In[36]:


# final data for model
model_df.info()


# ## Model

# In[37]:


# drop columns for final data for model
model_df = model_df.drop(["outletnumber"],axis=1)


# In[38]:


hrc_model = AgglomerativeClustering(n_clusters=5, linkage='ward', affinity='euclidean')
hrc_preds = hrc_model.fit_predict(model_df)


# In[39]:


model_df["hrc_cluster"] = hrc_preds


# In[40]:


df["hrc_cluster"] = hrc_preds


# In[41]:


model_df["hrc_cluster"].unique()


# In[42]:


model_df['hrc_cluster'].value_counts().plot(kind='pie')


# In[43]:


model_df['hrc_cluster'].value_counts()


# ## Decision Points

# In[44]:


from sklearn.model_selection import train_test_split

x = model_df.drop('hrc_cluster', axis=1)
y = model_df['hrc_cluster']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
dc = DecisionTreeClassifier(criterion="entropy", random_state=42, ) #max feat. bak bakalım
dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)

d_text = tree.export_text(dc)
print(d_text)


# In[45]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# In[46]:


dc.feature_importances_


# In[47]:


pd.Series(dc.feature_importances_, index=x_train.columns).nlargest(20).plot(kind='barh')


# In[48]:


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dc,
                   feature_names=model_data.columns,
                   filled=True)
for o in _:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(2)
        
plt.show()


# In[49]:


plt.figure(figsize=(15, 9))
sns.scatterplot(x="COCACOLA", y="SPRITE", hue=df['hrc_cluster'], data=df, palette=['green','red','dodgerblue',"orange","purple"])
plt.title('Hier. Clustering with 2 dimensions')


# In[50]:


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


# In[51]:


fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['SES'].replace({'A': 6, 'B': 5, 'C1': 4, 'C2': 3, 'D': 2, 'E': 1}), df['GECE_NUFUS'], df['GUNDUZ_NUFUS'],
                     c=hrc_preds, s=20, cmap="rainbow")


ax.set_title('Cluster Dist.')
ax.set_xlabel('SES')
ax.set_ylabel('GECE_NUFUS')
ax.set_zlabel('GUNDUZ_NUFUS')
plt.show()


# ## Results

# In[52]:


# extract coordinates
df["x"] = df["geometry"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[1])
df["y"] = df["geometry"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[0])


# In[53]:


df["y"] = pd.to_numeric(df["y"], downcast="float")
df["x"] = pd.to_numeric(df["x"], downcast="float")


# In[54]:


# Cluster separation
cluster_0 = df.query("hrc_cluster == 0")
cluster_1 = df.query("hrc_cluster == 1")
cluster_2 = df.query("hrc_cluster == 2")
cluster_3 = df.query("hrc_cluster == 3")
cluster_4 = df.query("hrc_cluster == 4")


# ### General Info.

# In[55]:


print("Toplam outlet sayısı : ", df.outletnumber.nunique())


# In[56]:


print("Totaldeki ortalama aylık harcama: ",df['AYLIK_HARCAMA'].mean())


# In[57]:


print("Totaldeki A,B,C1 kategorisindeki toplam outlet sayısı: ", df.query("SES == 'A' or SES == 'B' or SES == 'C1'").outletnumber.nunique())


# In[58]:


print("'Genç' kategorisi kapsamındaki toplam outlet sayısı: ", df.query("YAS_CLUSTER == 'YOUNG' or YAS_CLUSTER == 'TEEN' or YAS_CLUSTER == 'YOUNG ADULT'").outletnumber.nunique())


# &nbsp;

# ### Cluster-0

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama düşük.</dd> 
# <dd>2. SES kategorisi düşük.</dd>
# <dd>3. Gençler en yüksek.</dd>
# <dd>4. Yaşlı yok.</dd>
# <dd>5. Alkol/Tütün kullanımı yüksek.</dd>
# </font>

# In[59]:


print("Cluster-0'daki toplam outlet sayısı : ",cluster_0.outletnumber.nunique(), "\n")

print("Cluster-0'ın ortalama aylık harcaması: ",cluster_0['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-0'ın A,B,C1 kategorisindeki outlet oranı: ", '%' ,len(cluster_0.query("SES == 'A' or SES == 'B' or SES == 'C1'").outletnumber)/len(cluster_0.outletnumber)*100, "\n")

print("Cluster-0'ın 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_0.query("YAS_CLUSTER == 'YOUNG' or YAS_CLUSTER == 'TEEN' or YAS_CLUSTER == 'YOUNG ADULT'").outletnumber)/len(cluster_0.outletnumber)*100, "\n")

print("Cluster-0'ın 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_0.query("YAS_CLUSTER == 'OLD'").outletnumber)/len(cluster_0.outletnumber)*100, "\n")

print("Cluster-0'ın genel Alkol/Tütün oranı: ", '%', (cluster_0.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")


# &nbsp;

# ### Cluster-1

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama ortalama.</dd> 
# <dd>2. SES kategorisi epey yüksek.</dd>
# <dd>3. Gençler ve yaşlılar az.</dd>
# <dd>4. Orta yaş en fazla.</dd>
# <dd>5. Alkol/Tütün kullanımı yüksek.</dd>
# </font>

# In[60]:


print("Cluster-1'deki toplam outlet sayısı : ",cluster_1.outletnumber.nunique(), "\n")

print("Cluster-1'in ortalama aylık harcaması: ",cluster_1['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-1'in A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_1.query("SES == 'A' or SES == 'B' or SES == 'C1'").outletnumber)/len(cluster_1.outletnumber)*100, "\n")

print("Cluster-1'in 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_1.query("YAS_CLUSTER == 'YOUNG' or YAS_CLUSTER == 'TEEN' or YAS_CLUSTER == 'YOUNG ADULT'").outletnumber)/len(cluster_1.outletnumber)*100, "\n")

print("Cluster-1'in 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_1.query("YAS_CLUSTER == 'OLD'").outletnumber)/len(cluster_1.outletnumber)*100, "\n")

print("Cluster-1'in genel Alkol/Tütün oranı: ", '%', (cluster_1.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")


# &nbsp;

# ### Cluster-2

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama epey yüksek.</dd> 
# <dd>2. SES kategorisi yüksek.</dd>
# <dd>3. Gençler ve yaşlılar az.</dd>
# <dd>4. Orta yaş epey fazla.</dd>
# <dd>5. Alkol/Tütün kullanımı epey az.</dd>
# </font>

# In[61]:


print("Cluster-2'deki toplam outlet sayısı : ",cluster_2.outletnumber.nunique(), "\n")

print("Cluster-2'nin ortalama aylık harcaması: ",cluster_2['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-2'nin A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_2.query("SES == 'A' or SES == 'B' or SES == 'C1'").outletnumber)/len(cluster_2.outletnumber)*100, "\n")

print("Cluster-2'nin 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_2.query("YAS_CLUSTER == 'YOUNG' or YAS_CLUSTER == 'TEEN' or YAS_CLUSTER == 'YOUNG ADULT'").outletnumber)/len(cluster_2.outletnumber)*100, "\n")

print("Cluster-2'nin 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_2.query("YAS_CLUSTER == 'OLD'").outletnumber)/len(cluster_2.outletnumber)*100, "\n")

print("Cluster-2'nin genel Alkol/Tütün oranı: ", '%', (cluster_2.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")


# &nbsp;

# ### Cluster-3

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama en yüksek.</dd> 
# <dd>2. SES kategorisi en yüksek.</dd>
# <dd>3. Gençler az.</dd>
# <dd>4. Yaşlılar ve orta yaştakiler fazla.</dd>
# <dd>5. Alkol/Tütün kullanımı az.</dd>
# </font>

# In[62]:


print("Cluster-3'deki toplam outlet sayısı : ",cluster_3.outletnumber.nunique(), "\n")

print("Cluster-3'ün ortalama aylık harcaması: ",cluster_3['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-3'ün A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_3.query("SES == 'A' or SES == 'B' or SES == 'C1'").outletnumber)/len(cluster_3.outletnumber)*100, "\n")

print("Cluster-3'ün 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_3.query("YAS_CLUSTER == 'YOUNG' or YAS_CLUSTER == 'TEEN' or YAS_CLUSTER == 'YOUNG ADULT'").outletnumber)/len(cluster_3.outletnumber)*100, "\n")

print("Cluster-3'ün 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_3.query("YAS_CLUSTER == 'OLD'").outletnumber)/len(cluster_3.outletnumber)*100, "\n")

print("Cluster-3'ün genel Alkol/Tütün oranı: ", '%', (cluster_3.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")


# &nbsp;

# ### Cluster-4

# <font color='DarkSeaGreen'>  
# <dd>1. Aylık harcama orta.</dd> 
# <dd>2. SES kategorisi yüksek.</dd>
# <dd>3. Gençler az.</dd>
# <dd>4. Yaşlılar ve orta yaştakiler fazla.</dd>
# <dd>5. Alkol/Tütün kullanımı yok.</dd>
# </font>

# In[63]:


print("Cluster-4'deki toplam outlet sayısı : ",cluster_4.outletnumber.nunique(), "\n")

print("Cluster-4'ün ortalama aylık harcaması: ",cluster_4['AYLIK_HARCAMA'].mean(), "\n")

print("Cluster-4'ün A,B,C1 kategorisindeki outlet oranı: ",'%' ,len(cluster_4.query("SES == 'A' or SES == 'B' or SES == 'C1'").outletnumber)/len(cluster_4.outletnumber)*100, "\n")

print("Cluster-4'ün 'Genç' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_4.query("YAS_CLUSTER == 'YOUNG' or YAS_CLUSTER == 'TEEN' or YAS_CLUSTER == 'YOUNG ADULT'").outletnumber)/len(cluster_4.outletnumber)*100, "\n")

print("Cluster-4'ün 'Yaşlı' kategorisi kapsamındaki outlet yüzdesi: ", '%', len(cluster_4.query("YAS_CLUSTER == 'OLD'").outletnumber)/len(cluster_4.outletnumber)*100, "\n")

print("Cluster-4'ün genel Alkol/Tütün oranı: ", '%', (cluster_4.ALKOLTUTUN_ORAN.sum() / df.ALKOLTUTUN_ORAN.sum())*100, "\n")


# &nbsp;

# In[64]:


breakhere


# <h1><center>Cluster-0</center></h1>

# In[88]:


pie=cluster_0.groupby('YAS_CLUSTER').size().reset_index()
pie.columns=['YAS_CLUSTER','value']
px.pie(pie,values='value',names='YAS_CLUSTER', title='Age Dist. in CLUSTER-0')


# In[89]:


pie=cluster_0.groupby('SES').size().reset_index()
pie.columns=['SES','value']
px.pie(pie,values='value',names='SES', title='SES Dist. in CLUSTER-0')


# In[67]:


yas_brand_0 = cluster_0[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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
,'SPRITE']].groupby(['YAS_CLUSTER']).sum().reset_index()


# In[68]:


yas_brand_0


# In[69]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_0 = yas_brand_0.query("YAS_CLUSTER=='YOUNG'").groupby("YAS_CLUSTER").sum().T


# In[70]:


yas_cluster_0


# In[71]:


fig = px.pie(yas_cluster_0.reset_index(), values='YOUNG', names="index", title='Brand Dist. in Young Category (CLUSTER-0)')
fig.show()


# In[85]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_teen_0 = yas_brand_0.query("YAS_CLUSTER=='TEEN'").groupby("YAS_CLUSTER").sum().T


# In[87]:


fig = px.pie(yas_cluster_teen_0.reset_index(), values='TEEN', names="index", title='Brand Dist. in Teen Category (CLUSTER-0)')
fig.show()


# <h1><center>Cluster-1</center></h1>

# In[90]:


pie=cluster_1.groupby('YAS_CLUSTER').size().reset_index()
pie.columns=['YAS_CLUSTER','value']
px.pie(pie,values='value',names='YAS_CLUSTER', title='Age Dist. in CLUSTER-1')


# In[92]:


pie=cluster_1.groupby('SES').size().reset_index()
pie.columns=['SES','value']
px.pie(pie,values='value',names='SES', title='SES Dist. in CLUSTER-1')


# In[74]:


yas_brand_1 = cluster_1[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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
,'SPRITE']].groupby(['YAS_CLUSTER']).sum().reset_index()


# In[93]:


yas_cluster_1 = yas_brand_1.query("YAS_CLUSTER=='MIDDLE AGED'").groupby("YAS_CLUSTER").sum().T


# In[96]:


fig = px.pie(yas_cluster_1.reset_index(), values='MIDDLE AGED', names="index", title='Brand Dist. in Middle Aged Category (CLUSTER-1)')
fig.show()


# <h1><center>Cluster-2</center></h1>

# In[97]:


pie=cluster_2.groupby('YAS_CLUSTER').size().reset_index()
pie.columns=['YAS_CLUSTER','value']
px.pie(pie,values='value',names='YAS_CLUSTER', title='Age Dist. in CLUSTER-2')


# In[98]:


pie=cluster_2.groupby('SES').size().reset_index()
pie.columns=['SES','value']
px.pie(pie,values='value',names='SES', title='SES Dist. in CLUSTER-2')


# In[79]:


yas_brand_2 = cluster_2[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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
,'SPRITE']].groupby(['YAS_CLUSTER']).sum().reset_index()


# In[99]:


yas_brand_2


# In[115]:


yas_cluster_2 = yas_brand_2.query("YAS_CLUSTER=='MIDDLE AGED'").groupby("YAS_CLUSTER").sum().T


# In[116]:


yas_cluster_2


# In[105]:


yas_brand_2_sum = cluster_2[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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


# In[106]:


yas_brand_2_sum.rename({0: "value"}, inplace=True, axis=1)
yas_brand_2_sum = yas_brand_2_sum.iloc[1: , :]


# In[110]:


yas_brand_2_sum


# In[111]:


fig = px.pie(yas_brand_2_sum, values='value', names="index", title='Brand Dist. in CLUSTER-2')
fig.show()


# In[117]:


fig = px.pie(yas_cluster_2.reset_index(), values='MIDDLE AGED', names="index", title='Brand Dist. in Middle Aged Category (CLUSTER-2)')
fig.show()


# In[118]:


yas_cluster_2_old = yas_brand_2.query("YAS_CLUSTER=='OLD'").groupby("YAS_CLUSTER").sum().T


# In[119]:


yas_cluster_2_old


# In[120]:


fig = px.pie(yas_cluster_2_old.reset_index(), values='OLD', names="index", title='Brand Dist. in Old Category (CLUSTER-2)')
fig.show()


# <h1><center>Cluster-3</center></h1>

# In[121]:


pie=cluster_3.groupby('YAS_CLUSTER').size().reset_index()
pie.columns=['YAS_CLUSTER','value']
px.pie(pie,values='value',names='YAS_CLUSTER', title='Age Dist. in CLUSTER-3')


# In[122]:


pie=cluster_3.groupby('SES').size().reset_index()
pie.columns=['SES','value']
px.pie(pie,values='value',names='SES', title='SES Dist. in CLUSTER-3')


# In[124]:


yas_brand_3 = cluster_3[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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
,'SPRITE']].groupby(['YAS_CLUSTER']).sum().reset_index()


# In[125]:


yas_brand_3


# In[126]:


yas_cluster_3 = yas_brand_3.query("YAS_CLUSTER=='OLD'").groupby("YAS_CLUSTER").sum().T


# In[127]:


yas_cluster_3


# In[128]:


yas_brand_3_sum = cluster_3[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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


# In[129]:


yas_brand_3_sum.rename({0: "value"}, inplace=True, axis=1)
yas_brand_3_sum = yas_brand_3_sum.iloc[1: , :]


# In[130]:


yas_brand_3_sum


# In[131]:


fig = px.pie(yas_brand_3_sum, values='value', names="index", title='Brand Dist. in CLUSTER-3')
fig.show()


# In[132]:


fig = px.pie(yas_cluster_3.reset_index(), values='OLD', names="index", title='Brand Dist. in Old Category (CLUSTER-3)')
fig.show()


# In[134]:


yas_cluster_3_mid = yas_brand_3.query("YAS_CLUSTER=='MIDDLE AGED'").groupby("YAS_CLUSTER").sum().T


# In[137]:


fig = px.pie(yas_cluster_3_mid.reset_index(), values='MIDDLE AGED', names="index", title='Brand Dist. in Old Category (CLUSTER-3)')
fig.show()


# <h1><center>Cluster-4</center></h1>

# In[138]:


pie=cluster_4.groupby('YAS_CLUSTER').size().reset_index()
pie.columns=['YAS_CLUSTER','value']
px.pie(pie,values='value',names='YAS_CLUSTER', title='Age Dist. in CLUSTER-4')


# In[139]:


pie=cluster_4.groupby('SES').size().reset_index()
pie.columns=['SES','value']
px.pie(pie,values='value',names='SES', title='SES Dist. in CLUSTER-4')


# In[140]:


yas_brand_4 = cluster_4[['YAS_CLUSTER','BURN'                
,'CAPPY'               
,'CC_LIGHT'            
,'CC_NO_SUGER'         
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
,'SPRITE']].groupby(['YAS_CLUSTER']).sum().reset_index()


# In[141]:


yas_brand_4


# In[142]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_4 = yas_brand_4.query("YAS_CLUSTER=='OLD'").groupby("YAS_CLUSTER").sum().T


# In[144]:


fig = px.pie(yas_cluster_4.reset_index(), values='OLD', names="index", title='Brand Dist. in Old Category (CLUSTER-4)')
fig.show()


# In[146]:


# Hangi yaş grubundan bakmak istediğini burdan filtreleyebilirsin.
yas_cluster_mid_4 = yas_brand_4.query("YAS_CLUSTER=='MIDDLE AGED'").groupby("YAS_CLUSTER").sum().T


# In[147]:


fig = px.pie(yas_cluster_mid_4.reset_index(), values='MIDDLE AGED', names="index", title='Brand Dist. in Middle Age Category (CLUSTER-4)')
fig.show()


# ----

# In[ ]:


stophere


# In[ ]:


# bigquery'e tabloyu ekleme
# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

client = bigquery.Client()

table_id = "coca-cola-data-lake.predictive_order.shopper_profile_hier_clusters" 

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("outletname", bigquery.enums.SqlTypeNames.STRING),
    ],  write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(df, table_id, job_config=job_config)  
job.result()  

table = client.get_table(table_id)  
print("Loaded {} rows and {} columns to {}".format(table.num_rows, len(table.schema), table_id))


# In[ ]:




