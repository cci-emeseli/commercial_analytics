#!/usr/bin/env python
# coding: utf-8

# <h1><center>High Traffic Turkey Clustering</center></h1>

# Bu notebook ile yapılan çalışmada: 
# - High ve non-high trafik noktalarının belirlenmesi için **Türkiye genelinde** Traditional channel'da 2'li hierarchical clustering yapılmıştır
# - High ve non-high trafik noktalarının belirlenmesi için anket ve model sonuçlarının eşleştiği noktaları birebir alırken diğer kısımlarda classification yaparak max. tutarlılığı hedeflenmiştir. 
# 

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
import pickle
import sklearn
from sklearn import metrics
from numpy import mean
from numpy import std
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.svm import SVC
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
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('dark')


# In[2]:


bq_client = bigquery.Client()
bq_storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()

sql = """
select * 
from coca-cola-datalake-dev.DEV_EEMIRALI.SHOPPER_PROFILE
"""

shopper_df = bq_client.query(sql, location='EU').to_dataframe(bqstorage_client = bq_storage_client, progress_bar_type='tqdm')


# In[3]:


xls = pd.ExcelFile('../traffic_high_low.xlsx')
df_high = pd.read_excel(xls, 'High Traffic')
df_non = pd.read_excel(xls, 'Non-high Traffic')
all_data = pd.read_excel(xls, 'all_data')


# In[4]:


all_data.head()


# In[5]:


shopper_df.head(3)


# In[6]:


shopper_df = shopper_df.fillna(0)


# In[7]:


len(shopper_df)


# In[8]:


shopper_df.ILADI.value_counts()


# In[9]:


shopper_df.MAIN_CHANNEL.unique()


# ### Data Prep.

# In[10]:


model_data = shopper_df.query("MAIN_CHANNEL=='TRADITIONAL RETAIL'").reset_index(drop=True)


# In[11]:


len(model_data)


# In[12]:


model_data.drop_duplicates(keep = "first", inplace=True)


# In[13]:


len(model_data)


# In[14]:


model_df = model_data.copy()


# In[15]:


model_data.head(3)


# #### Label Encoding

# In[16]:


model_data["SES"] = model_data["SES"].replace({'A': 6, 'B': 5, 'C1': 4, 'C2': 3, 'D': 2, 'E': 1})


# In[17]:


model_data["SEGMENT"] = model_data["SEGMENT"].replace({'BRONZE': 1, 'SILVER PLUS': 2, 'SILVER': 3, 'GOLD': 4})


# #### İçeceklerin Oranları

# In[18]:


brand_list = ["BURN"
             ,"CAPPY"
             ,"CC_LIGHT"
             ,"COCACOLA"
             ,"COCACOLA_ENERGY"
             ,"DAMLA_MINERA"
             ,"DAMLA_WATER"
             ,"EXOTIC"
             ,"FANTA"
             ,"FUSETEA"
             ,"MONSTER"
             ,"POWERADE"
             ,"SCHWEPPES"
             ,"SPRITE"]


# In[19]:


m = model_data[brand_list]


# In[20]:


m['total'] = m.sum(axis=1)


# In[21]:


m.head(3)


# In[22]:


model_data["BURN"] = m["BURN"]/m["total"]
model_data["CAPPY"] = m["CAPPY"]/m["total"]
model_data["CC_LIGHT"] = m["CC_LIGHT"]/m["total"]
model_data["COCACOLA"] = m["COCACOLA"]/m["total"]
model_data["COCACOLA_ENERGY"] = m["COCACOLA_ENERGY"]/m["total"]
model_data["DAMLA_MINERA"] = m["DAMLA_MINERA"]/m["total"]
model_data["DAMLA_WATER"] = m["DAMLA_WATER"]/m["total"]
model_data["EXOTIC"] = m["EXOTIC"]/m["total"]
model_data["FANTA"] = m["FANTA"]/m["total"]
model_data["FUSETEA"] = m["FUSETEA"]/m["total"]
model_data["MONSTER"] = m["MONSTER"]/m["total"]
model_data["POWERADE"] = m["POWERADE"]/m["total"]
model_data["SCHWEPPES"] = m["SCHWEPPES"]/m["total"]
model_data["SPRITE"] = m["SPRITE"]/m["total"]


# #### Gece Gündüz Oranı

# In[23]:


# drop columns for final data for model
model_data = model_data.drop(["outlet_name"                   
                             ,"GEOGPOINT"
                             ,"IDARIID"
                             ,"MAIN_CHANNEL"
                             ,"YAS_CLUSTER"
                             ,"ILADI"
                             ,"ILCEADI"
                             ,"HANE_BUYUKLUGU"
                             ,"ZENGINLIK_INDEKSI"
                             ],axis=1)


# In[24]:


model_data['GECE_NUFUS'].replace({0: 0.00001}, inplace=True)
model_data['GUNDUZ_NUFUS'].replace({0: 0.00001}, inplace=True)


# In[25]:


model_data.head(3)


# In[26]:


g = model_data[['GUNDUZ_NUFUS','GECE_NUFUS']]


# In[27]:


model_data["gündüz_gece_oran"] = g["GUNDUZ_NUFUS"]/g["GECE_NUFUS"]


# In[28]:


model_data.head(3)


# In[29]:


model_data = model_data.fillna(0)


# In[30]:


model_data['SES'] = model_data['SES'].astype(str).astype(int)


# In[31]:


model_data.SES.unique()


# In[32]:


model_data.info()


# #### Scaling and Normal Dist.

# In[33]:


# define a method to scale data, looping thru the columns, and passing a scaler
def scale_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in data.columns:
        data[col] = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# In[34]:


def normal_dist(data):
    for col in data.columns:
        data[col] = data[col].apply(lambda x: boxcox1p(x,0.25))
        stats.boxcox(data[col])[0]
    return data


# In[35]:


def log_data(data):
    for col in data.columns:
        data[col] = data[col].apply(lambda x: np.log(x))
    return data


# In[36]:


copy_data = model_data.copy()


# In[37]:


copy_data = copy_data.drop(["outlet_number"                   
                            ,'GUNDUZ_NUFUS'
                            ,'GECE_NUFUS'],axis=1)


# In[38]:


copy_data.head(2)


# In[39]:


copy_data = copy_data.drop(["IL_BAZLI_TURIST_YERLI_ISLETME","IL_BAZLI_TURIST_YBN_ISLETME"],axis=1)


# In[40]:


copy_data.head(3)


# In[41]:


copy_data[copy_data < 0] = 0


# In[42]:


copy_data.replace({0: 0.00001}, inplace=True)


# #### Normal Dist. Part 

# In[43]:


copy_data["SES"] = stats.boxcox(copy_data["SES"])[0]
copy_data["SEHIRLESME_INDEKSI"] = stats.boxcox(copy_data["SEHIRLESME_INDEKSI"])[0]
copy_data["YAYA_TRAFIGI"] = stats.boxcox(copy_data["YAYA_TRAFIGI"])[0]
copy_data["AYLIK_HARCAMA"] = stats.boxcox(copy_data["AYLIK_HARCAMA"])[0]
copy_data["GIDAVEICECEK"] = stats.boxcox(copy_data["GIDAVEICECEK"])[0]
copy_data["ALKOLTUTUN"] = stats.boxcox(copy_data["ALKOLTUTUN"])[0]
copy_data["EGLENCEKULTUR"] = stats.boxcox(copy_data["EGLENCEKULTUR"])[0]
copy_data["LOKANTAOTEL"] = stats.boxcox(copy_data["LOKANTAOTEL"])[0]
copy_data["GIDAVEICECEK_ORAN"] = stats.boxcox(copy_data["GIDAVEICECEK_ORAN"])[0]
copy_data["ALKOLTUTUN_ORAN"] = stats.boxcox(copy_data["ALKOLTUTUN_ORAN"])[0]
copy_data["EGLENCEKULTUR_ORAN"] = stats.boxcox(copy_data["EGLENCEKULTUR_ORAN"])[0]
copy_data["LOKANTAOTEL_ORAN"] = stats.boxcox(copy_data["LOKANTAOTEL_ORAN"])[0]
copy_data["ISYERI_YOGUNLUGU_SAYI_KM2_YERLE"] = stats.boxcox(copy_data["ISYERI_YOGUNLUGU_SAYI_KM2_YERLE"])[0]
copy_data["ISYERI_YOGUNLUGU_SAYI_KM2"] = stats.boxcox(copy_data["ISYERI_YOGUNLUGU_SAYI_KM2"])[0]
copy_data["KONUT_YOGUNLUGU_SAYI_KM2_YERLES"] = stats.boxcox(copy_data["KONUT_YOGUNLUGU_SAYI_KM2_YERLES"])[0]
copy_data["KONUT_YOGUNLUGU_SAYI_KM2"] = stats.boxcox(copy_data["KONUT_YOGUNLUGU_SAYI_KM2"])[0]
#copy_data["IL_BAZLI_TURIST_YERLI_ISLETME"] = stats.boxcox(copy_data["IL_BAZLI_TURIST_YERLI_ISLETME"])[0]
#copy_data["IL_BAZLI_TURIST_YBN_ISLETME"] = stats.boxcox(copy_data["IL_BAZLI_TURIST_YBN_ISLETME"])[0]
copy_data["ORTA_ALT"] = stats.boxcox(copy_data["ORTA_ALT"])[0]
copy_data["ORTA"] = stats.boxcox(copy_data["ORTA"])[0]
copy_data["ORTA_UST"] = stats.boxcox(copy_data["ORTA_UST"])[0]
copy_data["UST"] = stats.boxcox(copy_data["UST"])[0]
copy_data["AKARYAKIT"] = stats.boxcox(copy_data["AKARYAKIT"])[0]
copy_data["ALISVERIS"] = stats.boxcox(copy_data["ALISVERIS"])[0]
copy_data["OUTLET"] = stats.boxcox(copy_data["OUTLET"])[0]
copy_data["KONUT"] = stats.boxcox(copy_data["KONUT"])[0]
copy_data["EGITIM"] = stats.boxcox(copy_data["EGITIM"])[0]
copy_data["UNIVERSITE"] = stats.boxcox(copy_data["UNIVERSITE"])[0]
copy_data["EGLENCE"] = stats.boxcox(copy_data["EGLENCE"])[0]
copy_data["KULTUREL"] = stats.boxcox(copy_data["KULTUREL"])[0]
copy_data["TURIZM"] = stats.boxcox(copy_data["TURIZM"])[0]
copy_data["SPOR"] = stats.boxcox(copy_data["SPOR"])[0]
copy_data["HASTANE"] = stats.boxcox(copy_data["HASTANE"])[0]
copy_data["ASKERI"] = stats.boxcox(copy_data["ASKERI"])[0]
copy_data["DEMIRYOLU"] = stats.boxcox(copy_data["DEMIRYOLU"])[0]
copy_data["DENIZYOLU"] = stats.boxcox(copy_data["DENIZYOLU"])[0]
copy_data["HAVAYOLU"] = stats.boxcox(copy_data["HAVAYOLU"])[0]
copy_data["KARAYOLU"] = stats.boxcox(copy_data["KARAYOLU"])[0]
copy_data["TRAFIK"] = stats.boxcox(copy_data["TRAFIK"])[0]
copy_data["E_COMMERCE"] = stats.boxcox(copy_data["E_COMMERCE"])[0]
copy_data["MODERN_RETAIL"] = stats.boxcox(copy_data["MODERN_RETAIL"])[0]
copy_data["ON_PREMISE"] = stats.boxcox(copy_data["ON_PREMISE"])[0]
copy_data["OTHER_THIRD_PARTY"] = stats.boxcox(copy_data["OTHER_THIRD_PARTY"])[0]
copy_data["TRADITIONAL_RETAIL"] = stats.boxcox(copy_data["TRADITIONAL_RETAIL"])[0]
copy_data["BURN"] = stats.boxcox(copy_data["BURN"])[0]
copy_data["CAPPY"] = stats.boxcox(copy_data["CAPPY"])[0]
copy_data["CC_LIGHT"] = stats.boxcox(copy_data["CC_LIGHT"])[0]
copy_data["COCACOLA"] = stats.boxcox(copy_data["COCACOLA"])[0]
copy_data["COCACOLA_ENERGY"] = stats.boxcox(copy_data["COCACOLA_ENERGY"])[0]
copy_data["DAMLA_MINERA"] = stats.boxcox(copy_data["DAMLA_MINERA"])[0]
copy_data["DAMLA_WATER"] = stats.boxcox(copy_data["DAMLA_WATER"])[0]
copy_data["EXOTIC"] = stats.boxcox(copy_data["EXOTIC"])[0]
copy_data["FANTA"] = stats.boxcox(copy_data["FANTA"])[0]
copy_data["FUSETEA"] = stats.boxcox(copy_data["FUSETEA"])[0]
copy_data["MONSTER"] = stats.boxcox(copy_data["MONSTER"])[0]
copy_data["POWERADE"] = stats.boxcox(copy_data["POWERADE"])[0]
copy_data["SCHWEPPES"] = stats.boxcox(copy_data["SCHWEPPES"])[0]
copy_data["SPRITE"] = stats.boxcox(copy_data["SPRITE"])[0]


# In[44]:


sns.displot(model_data['AYLIK_HARCAMA'])


# In[45]:


sns.displot(copy_data["AYLIK_HARCAMA"])


# In[46]:


sns.displot(model_data["LOKANTAOTEL"])


# In[47]:


sns.displot(copy_data["LOKANTAOTEL"])


# In[48]:


model_data.SES.hist()


# In[49]:


copy_data.SES.hist()


# #### Scaling Part

# In[50]:


scale_df = copy_data.copy()


# In[51]:


scale_df = scale_data(scale_df)


# In[52]:


sns.displot(copy_data['AYLIK_HARCAMA'])


# In[53]:


sns.displot(scale_df["AYLIK_HARCAMA"])


# # Model 

# In[54]:


scale_df.isna().sum().sum()


# ### K-Means

# In[55]:


km_model = KMeans(n_clusters=2, random_state=42, init='k-means++')
km_model.fit(scale_df)
km_preds = km_model.predict(scale_df)


# In[56]:


model_df["preds"] = km_preds


# In[57]:


model_df


# ## Decision Points

# In[58]:


# decision tree için kullanılacak bu data; içinde scale edilmiş değerler yanında hier. clustering prediction'larını da içerir.
decision_df = scale_df.copy()


# In[59]:


decision_df['preds'] = km_preds


# In[60]:


x = decision_df.drop('preds', axis=1)
y = decision_df['preds']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=41)
    
dc = DecisionTreeClassifier(criterion="entropy", random_state=42) 
dc.fit(x_train, y_train)
dc_pred = dc.predict(x_test)

d_text = tree.export_text(dc)
print(d_text)


# In[61]:


metrics.accuracy_score(y_test, dc_pred)


# In[62]:


dc.feature_importances_


# In[63]:


pd.Series(dc.feature_importances_, index=x_train.columns).nlargest(15).plot(kind='barh')


# In[64]:


final_high = model_df.query("preds == 1")
final_low = model_df.query("preds == 0") 


# ## High Traffic

# In[65]:


print("High Traffic çıkan toplam outlet sayısı : ", final_high.outlet_number.nunique(), "\n")

print("High Traffic'in ortalama aylık harcaması: ", final_high['AYLIK_HARCAMA'].mean(), "\n") 

print("High Traffic'in ortalama yaya trafiği: ", final_high['YAYA_TRAFIGI'].mean(), "\n") 

print("High Traffic'in A,B,C1 kategorisindeki toplam outlet sayısı oranı: ", "%",final_high.query("SES == 'A' or SES == 'B' or SES == 'C1'").outlet_number.nunique()/len(final_high.SES) * 100  , "\n")

print("High Traffic'in ortalama Gündüz Nufusu sayısı: ", final_high.GUNDUZ_NUFUS.mean(), "\n")

print("High Traffic'in ortalama Gece Nufusu sayısı: ", final_high.GECE_NUFUS.mean(), "\n")

print("High Traffic'in ortalama Eğitim birimi sayısı: ", final_high.EGITIM.mean())
print("High Traffic'in minimum Eğitim birimi sayısı: ", final_high.EGITIM.min())
print("High Traffic'in maximum Eğitim birimi sayısı: ", final_high.EGITIM.max(), "\n")

print("High Traffic'in ortalama Modern Retail birimi sayısı: ", final_high.MODERN_RETAIL.mean(), "\n")

print("High Traffic'in ortalama Spor birimi sayısı: ", final_high.SPOR.mean(), "\n")


# <br>

# #### Non-high Traffic

# In[66]:


print("Non-high Traffic çıkan toplam outlet sayısı : ", final_low.outlet_number.nunique(), "\n")

print("Non-high Traffic'in ortalama aylık harcaması: ", final_low['AYLIK_HARCAMA'].mean(), "\n") 

print("Non-high Traffic'in ortalama yaya trafiği: ", final_low['YAYA_TRAFIGI'].mean(), "\n") 

print("Non-high Traffic'in A,B,C1 kategorisindeki toplam outlet sayısı oranı: ", "%", final_low.query("SES == 'A' or SES == 'B' or SES == 'C1'").outlet_number.nunique()/len(final_low.SES) * 100  , "\n")

print("Non-high Traffic'in ortalama Gündüz Nufusu sayısı: ", final_low.GUNDUZ_NUFUS.mean(), "\n")

print("Non-high Traffic'in ortalama Gece Nufusu sayısı: ", final_low.GECE_NUFUS.mean(), "\n")

print("Non-high'in ortalama Eğitim birimi sayısı: ", final_low.EGITIM.mean())
print("Non-high'in minimum Eğitim birimi sayısı: ", final_low.EGITIM.min())
print("Non-high'in maximum Eğitim birimi sayısı: ", final_low.EGITIM.max(), "\n")

print("Non-high'in ortalama Modern Retail birimi sayısı: ", final_low.MODERN_RETAIL.mean(), "\n")

print("Non-high'in ortalama Spor birimi sayısı: ", final_low.SPOR.mean(), "\n")


# ## Anket Sonrası Karşılaştırma

# In[67]:


anket = pd.read_csv("data/anket.csv")


# In[68]:


anket["outlet_number"] = anket["outlet_number"].astype(str)


# In[69]:


anket.info()


# In[70]:


model_df["outlet_number"] = model_df["outlet_number"].astype(str)


# In[71]:


merge_df = pd.merge(model_df, anket[["outlet_number","result"]], on='outlet_number', how='inner').reset_index(drop=True)


# In[72]:


merge_df.drop_duplicates(keep = "first", inplace=True)


# In[73]:


len(merge_df)


# - Anket ve shopper_profile datasının ortak 131269 outleti var. (duplicate olanlar atılmış bir biçimde)

# In[74]:


merge_df.head(3)


# In[75]:


pd.Series(dc.feature_importances_, index=x_train.columns).nlargest(15).plot(kind='barh')


# In[76]:


print("total data lenght is:", len(merge_df)) 


# In[77]:


print("Non-high is:", len(merge_df.loc[(merge_df['preds'] == 0) & (merge_df['result'] == 0)]))


# In[78]:


print("High-Traffic is:", len(merge_df.loc[(merge_df['preds'] == 1) & (merge_df['result'] == 1)]))


# In[79]:


print("1-0:", len(merge_df.loc[(merge_df['preds'] == 1) & (merge_df['result'] == 0)]))


# In[80]:


print("0-1:", len(merge_df.loc[(merge_df['preds'] == 0) & (merge_df['result'] == 1)]))


# In[81]:


print("Accuracy rate: ", (len(merge_df.loc[(merge_df['preds'] == 0) & (merge_df['result'] == 0)]) + len(merge_df.loc[(merge_df['preds'] == 1) & (merge_df['result'] == 1)])) / len(merge_df))


# ### İstanbul

# In[82]:


ist_df = merge_df.query("ILADI == 'İstanbul'")


# In[83]:


print("total data lenght is:", len(ist_df)) 


# In[84]:


print("Non-high is:", len(ist_df.loc[(ist_df['preds'] == 0) & (ist_df['result'] == 0)]))


# In[85]:


print("High-Traffic is:", len(ist_df.loc[(ist_df['preds'] == 1) & (ist_df['result'] == 1)]))


# In[86]:


print("1-0:", len(ist_df.loc[(ist_df['preds'] == 1) & (ist_df['result'] == 0)]))


# In[87]:


print("0-1:", len(ist_df.loc[(ist_df['preds'] == 0) & (ist_df['result'] == 1)]))


# In[88]:


print("Accuracy rate: ", (len(ist_df.loc[(ist_df['preds'] == 0) & (ist_df['result'] == 0)]) + len(ist_df.loc[(ist_df['preds'] == 1) & (ist_df['result'] == 1)])) / len(ist_df))


# ## Classification

# In[89]:


# Bunlar mutlak doğru olarak kabul edildiğinden bir kenarda dursun.
high_df = merge_df.loc[(merge_df['preds'] == 0) & (merge_df['result'] == 0)]
non_high_df = merge_df.loc[(merge_df['preds'] == 1) & (merge_df['result'] == 1)]


# In[90]:


# Geriye bunlar kaldı, bunlar üzerinden de classification yapılarak en sonunda bütün veriler birleştirilecek.
one_zero_df = merge_df.loc[(merge_df['preds'] == 1) & (merge_df['result'] == 0)]
zero_one_df = merge_df.loc[(merge_df['preds'] == 0) & (merge_df['result'] == 1)]


# In[91]:


test_df = pd.concat([one_zero_df, zero_one_df]).reset_index(drop=True).drop('preds', axis=1)


# In[92]:


# classification'da test datası bu df olacak.
test_df.head(3)


# In[93]:


train_df = pd.concat([high_df, non_high_df]).reset_index(drop=True).drop('preds', axis=1)


# In[94]:


# classification'da test datası bu df olacak.
train_df.head(3)


# In[95]:


len(train_df)


# In[96]:


len(test_df)


# In[97]:


print(len(test_df)+len(train_df))


# ### Data Prep.

# In[98]:


onumber_list = train_df.outlet_number.tolist()


# In[99]:


# zaten üstte data prep. aşamasından geçmiş model_data dataframe'ini elimdeki train_df'in outlet_number ile eşleştirip --
# -- yanına result column'u da ekleyerek modele hazır hale getiriyorum.

# Amacım önce train_df üzerinden bir classification modeli çıkartmak ve daha sonrasında test_df'i predict ettirmek. 

train_data = model_data[model_data.isin(onumber_list).any(1).values].reset_index(drop=True)
train_data = pd.merge(train_data, train_df[["outlet_number","result"]], on='outlet_number', how='inner').reset_index(drop=True)


# In[100]:


train_data.head(3)


# In[101]:


del train_data["outlet_number"]


# In[102]:


#split dataset in features and target variable
X = train_data.drop('result', axis=1) # Features
y = train_data["result"] # Target variable


# In[103]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# ## Classification Model 

# In[104]:


#Let's create a Decision Tree Model using Scikit-learn.

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[105]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# ## Predict test data 

# ### Data Prep.

# In[106]:


onumber_list2 = test_df.outlet_number.tolist()


# In[107]:


# zaten üstte data prep. aşamasından geçmiş model_data dataframe'ini elimdeki test_df'in outlet_number ile eşleştirip --
# -- yanına result column'u da ekleyerek modele hazır hale getiriyorum.

# train_df üzerinden bir classification modeli çıkartmıştım şimdi test_df'i predict ettircem. 

test_data = model_data[model_data.isin(onumber_list2).any(1).values].reset_index(drop=True)
test_data = pd.merge(test_data, test_df[["outlet_number","result"]], on='outlet_number', how='inner').reset_index(drop=True)


# In[108]:


test_data.head(3)


# In[109]:


test_copy_df = test_data.drop(['outlet_number','result'], axis=1)


# In[110]:


test_copy_df.head(2)


# In[111]:


dc_preds = clf.predict(test_copy_df)


# In[112]:


test_data["preds"] = dc_preds


# In[113]:


test_data.head()


# In[114]:


print("total data lenght is:", len(test_data)) 


# In[115]:


print("Non-high is:", len(test_data.loc[(test_data['preds'] == 0) & (test_data['result'] == 0)]))


# In[116]:


print("High-Traffic is:", len(test_data.loc[(test_data['preds'] == 1) & (test_data['result'] == 1)]))


# In[117]:


print("1-0:", len(test_data.loc[(test_data['preds'] == 1) & (test_data['result'] == 0)]))


# In[118]:


print("0-1:", len(test_data.loc[(test_data['preds'] == 0) & (test_data['result'] == 1)]))


# In[119]:


print("Accuracy rate: ", (len(test_data.loc[(test_data['preds'] == 0) & (test_data['result'] == 0)]) + len(test_data.loc[(test_data['preds'] == 1) & (test_data['result'] == 1)])) / len(test_data))


# ## Decision Points of DecisionTreeClassifier

# In[220]:


decision2_df = test_data.drop(['outlet_number','result'], axis=1)


# In[224]:


decision2_df


# In[225]:


x = decision2_df.drop('preds', axis=1)
y = decision2_df['preds']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=41)
    
dc2 = DecisionTreeClassifier(criterion="entropy", random_state=42) 
dc2.fit(x_train, y_train)
dc_pred2 = dc2.predict(x_test)

d_text2 = tree.export_text(dc)
print(d_text2)


# In[226]:


metrics.accuracy_score(y_test, dc_pred2)


# In[227]:


dc2.feature_importances_


# In[228]:


pd.Series(dc2.feature_importances_, index=x_train.columns).nlargest(15).plot(kind='barh')


# ------

# In[120]:


# geriye bütün dataları alt alta eklemek kaldı
final_test_df = pd.merge(test_df, test_data[["outlet_number", "preds"]], on='outlet_number', how='inner').reset_index(drop=True)


# In[121]:


len(final_test_df)


# In[131]:


frames = [high_df, non_high_df, final_test_df]
result_df = pd.concat(frames).reset_index(drop=True)


# In[132]:


del result_df["result"]


# In[133]:


result_df.head(3)


# In[134]:


len(result_df)


# In[135]:


result_df.isna().sum().sum()


# In[144]:


result_df.drop_duplicates(subset="outlet_number", keep="first", inplace=True)


# In[145]:


len(result_df)


# ## High Traffic

# In[146]:


nh_df = result_df.query("preds == 0")
h_df = result_df.query("preds == 1")


# In[211]:


print("High Traffic çıkan toplam outlet sayısı : ", h_df.outlet_number.nunique(), "\n")

print("High Traffic'in ortalama aylık harcaması: ", h_df['AYLIK_HARCAMA'].mean(), "\n") 

print("High Traffic'in ortalama yaya trafiği: ", h_df['YAYA_TRAFIGI'].mean(), "\n") 

print("High Traffic'in A,B,C1 kategorisindeki toplam outlet sayısı oranı: ", "%",h_df.query("SES == 'A' or SES == 'B' or SES == 'C1'").outlet_number.nunique()/len(h_df.SES) * 100  , "\n")

print("High Traffic'in ortalama Gündüz Nufusu sayısı: ", h_df.GUNDUZ_NUFUS.mean(), "\n")

print("High Traffic'in ortalama Gece Nufusu sayısı: ", h_df.GECE_NUFUS.mean(), "\n")

print("High Traffic'in ortalama Eğitim birimi sayısı: ", h_df.EGITIM.mean())
print("High Traffic'in minimum Eğitim birimi sayısı: ", h_df.EGITIM.min())
print("High Traffic'in maximum Eğitim birimi sayısı: ", h_df.EGITIM.max(), "\n")

print("High Traffic'in ortalama Modern Retail birimi sayısı: ", h_df.MODERN_RETAIL.mean(), "\n")

print("High Traffic'in ortalama Spor birimi sayısı: ", h_df.SPOR.mean(), "\n")


# <br>

# ## Non-high Traffic

# In[213]:


print("Non-high Traffic çıkan toplam outlet sayısı : ", nh_df.outlet_number.nunique(), "\n")

print("Non-high Traffic'in ortalama aylık harcaması: ", nh_df['AYLIK_HARCAMA'].mean(), "\n") 

print("Non-high Traffic'in ortalama yaya trafiği: ", nh_df['YAYA_TRAFIGI'].mean(), "\n") 

print("Non-high Traffic'in A,B,C1 kategorisindeki toplam outlet sayısı oranı: ", "%", nh_df.query("SES == 'A' or SES == 'B' or SES == 'C1'").outlet_number.nunique()/len(nh_df.SES) * 100  , "\n")

print("Non-high Traffic'in ortalama Gündüz Nufusu sayısı: ", nh_df.GUNDUZ_NUFUS.mean(), "\n")

print("Non-high Traffic'in ortalama Gece Nufusu sayısı: ", nh_df.GECE_NUFUS.mean(), "\n")

print("Non-high'in ortalama Eğitim birimi sayısı: ", nh_df.EGITIM.mean())
print("Non-high'in minimum Eğitim birimi sayısı: ", nh_df.EGITIM.min())
print("Non-high'in maximum Eğitim birimi sayısı: ", nh_df.EGITIM.max(), "\n")

print("Non-high'in ortalama Modern Retail birimi sayısı: ", nh_df.MODERN_RETAIL.mean(), "\n")

print("Non-high'in ortalama Spor birimi sayısı: ", nh_df.SPOR.mean(), "\n")


# ### İstanbul

# In[149]:


ist_df2 = final_test_df.query("ILADI == 'İstanbul'")


# In[150]:


print("total data lenght is:", len(ist_df2)) 


# In[151]:


print("Non-high is:", len(ist_df2.loc[(ist_df2['preds'] == 0) & (ist_df2['result'] == 0)]))


# In[152]:


print("High-Traffic is:", len(ist_df2.loc[(ist_df2['preds'] == 1) & (ist_df2['result'] == 1)]))


# In[153]:


print("1-0:", len(ist_df2.loc[(ist_df2['preds'] == 1) & (ist_df2['result'] == 0)]))


# In[154]:


print("0-1:", len(ist_df2.loc[(ist_df2['preds'] == 0) & (ist_df2['result'] == 1)]))


# In[155]:


print("Accuracy rate: ", (len(ist_df2.loc[(ist_df2['preds'] == 0) & (ist_df2['result'] == 0)]) + len(ist_df2.loc[(ist_df2['preds'] == 1) & (ist_df2['result'] == 1)])) / len(ist_df2))


# In[156]:


result_df.head(3)


# In[157]:


# extract coordinates
result_df["x"] = result_df["GEOGPOINT"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[1])
result_df["y"] = result_df["GEOGPOINT"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[0])


# In[158]:


result_df["y"] = pd.to_numeric(result_df["y"], downcast="float")
result_df["x"] = pd.to_numeric(result_df["x"], downcast="float")

# bigquery'e tabloyu ekleme
# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

client = bigquery.Client()

table_id = "coca-cola-data-lake.predictive_order.high_traffic_anket_sonrasi_general" 

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("outlet_name", bigquery.enums.SqlTypeNames.STRING),
    ],  write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(result_df, table_id, job_config=job_config)  
job.result()  

table = client.get_table(table_id)  
print("Loaded {} rows and {} columns to {}".format(table.num_rows, len(table.schema), table_id))
# In[232]:


result_df.head()


# ---

# ## Türkiye Genelinde Anket vs General Model

# In[185]:


# geriye bütün dataları alt alta eklemek kaldı
anket_vs_df = pd.merge(result_df, anket[["outlet_number", "result"]], on='outlet_number', how='inner').reset_index(drop=True)


# In[183]:


len(anket_vs_df)


# In[187]:


anket_vs_df.drop_duplicates(keep = "first", inplace=True)


# In[208]:


len(anket_vs_df)


# In[188]:


print("total data lenght is:", len(anket_vs_df)) 


# In[189]:


print("Non-high is:", len(anket_vs_df.loc[(anket_vs_df['preds'] == 0) & (anket_vs_df['result'] == 0)]))


# In[190]:


print("High-Traffic is:", len(anket_vs_df.loc[(anket_vs_df['preds'] == 1) & (anket_vs_df['result'] == 1)]))


# In[191]:


print("1-0:", len(anket_vs_df.loc[(anket_vs_df['preds'] == 1) & (anket_vs_df['result'] == 0)]))


# In[192]:


print("0-1:", len(anket_vs_df.loc[(anket_vs_df['preds'] == 0) & (anket_vs_df['result'] == 1)]))


# In[193]:


print("Accuracy rate: ", (len(anket_vs_df.loc[(anket_vs_df['preds'] == 0) & (anket_vs_df['result'] == 0)]) + len(anket_vs_df.loc[(anket_vs_df['preds'] == 1) & (anket_vs_df['result'] == 1)])) / len(anket_vs_df))


# ## İstanbul Genelinde Anket vs General Model

# In[194]:


anket_vs_ist = anket_vs_df.query("ILADI == 'İstanbul'").reset_index(drop=True)


# In[195]:


anket_vs_ist.head(3)


# In[196]:


print("total data lenght is:", len(anket_vs_ist)) 


# In[197]:


print("Non-high is:", len(anket_vs_ist.loc[(anket_vs_ist['preds'] == 0) & (anket_vs_ist['result'] == 0)]))


# In[198]:


print("High-Traffic is:", len(anket_vs_ist.loc[(anket_vs_ist['preds'] == 1) & (anket_vs_ist['result'] == 1)]))


# In[199]:


print("1-0:", len(anket_vs_ist.loc[(anket_vs_ist['preds'] == 1) & (anket_vs_ist['result'] == 0)]))


# In[200]:


print("0-1:", len(anket_vs_ist.loc[(anket_vs_ist['preds'] == 0) & (anket_vs_ist['result'] == 1)]))


# In[201]:


print("Accuracy rate: ", (len(anket_vs_ist.loc[(anket_vs_ist['preds'] == 0) & (anket_vs_ist['result'] == 0)]) + len(anket_vs_ist.loc[(anket_vs_ist['preds'] == 1) & (anket_vs_ist['result'] == 1)])) / len(anket_vs_ist))


# ----

# ### General Cluster Model Çıktılarının Superset'e Aktarılması

# In[202]:


anket_vs_df.head(3)


# In[203]:


## FOR TURKIYE ## 

non_high= anket_vs_df.loc[(anket_vs_df['preds'] == 0) & (anket_vs_df['result'] == 0)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
non_high["comparison"] = 1

high = anket_vs_df.loc[(anket_vs_df['preds'] == 1) & (anket_vs_df['result'] == 1)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
high["comparison"] = 2

one_zero = anket_vs_df.loc[(anket_vs_df['preds'] == 1) & (anket_vs_df['result'] == 0)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
one_zero["comparison"] = 3

zero_one = anket_vs_df.loc[(anket_vs_df['preds'] == 0) & (anket_vs_df['result'] == 1)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
zero_one["comparison"] = 4


# In[204]:


frames = [non_high, high, one_zero, zero_one]
superset_df = pd.concat(frames).sample(frac=1).reset_index(drop=True)


# In[205]:


superset_final = pd.merge(anket_vs_df, superset_df[["outlet_number","comparison"]], on='outlet_number', how='inner').reset_index(drop=True)

## FOR ISTANBUL ## 

non_high= anket_vs_ist.loc[(anket_vs_ist['preds'] == 0) & (anket_vs_ist['result'] == 0)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
non_high["comparison"] = 1

high = anket_vs_ist.loc[(anket_vs_ist['preds'] == 1) & (anket_vs_ist['result'] == 1)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
high["comparison"] = 2

one_zero = anket_vs_ist.loc[(anket_vs_ist['preds'] == 1) & (anket_vs_ist['result'] == 0)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
one_zero["comparison"] = 3

zero_one = anket_vs_ist.loc[(anket_vs_ist['preds'] == 0) & (anket_vs_ist['result'] == 1)].reset_index(drop=True)[["outlet_number","outlet_name","x","y"]]
zero_one["comparison"] = 4frames = [non_high, high, one_zero, zero_one]
superset_df = pd.concat(frames).sample(frac=1).reset_index(drop=True)
# In[206]:


superset_final.head(3)


# In[207]:


stophere


# In[209]:


# bigquery'e tabloyu ekleme
# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

client = bigquery.Client()

table_id = "coca-cola-data-lake.predictive_order.high_traffic_anket_vs_model_turkiye" 

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("outlet_name", bigquery.enums.SqlTypeNames.STRING),
    ],  write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(superset_final, table_id, job_config=job_config)  
job.result()  

table = client.get_table(table_id)  
print("Loaded {} rows and {} columns to {}".format(table.num_rows, len(table.schema), table_id))


# In[230]:


superset_final.to_csv("model_a.csv")

