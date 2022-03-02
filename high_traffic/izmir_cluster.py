#!/usr/bin/env python
# coding: utf-8

# <h1><center>High Traffic Clustering</center></h1>

# Bu notebook, high ve non-high trafik noktalarının belirlenmesi içi **İZMİR** özelinde Traditional channel'da 2'li hierarchical clustering çalışmasıdır.

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

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('dark')


# In[2]:


xls = pd.ExcelFile('traffic_high_low.xlsx')
df_high = pd.read_excel(xls, 'High Traffic')
df_non = pd.read_excel(xls, 'Non-high Traffic')
all_data = pd.read_excel(xls, 'all_data')


# In[3]:


all_data.head()


# In[15]:


df_new = pd.read_csv('shopper_data.csv')


# In[16]:


df_new = df_new.fillna(0)


# In[17]:


len(df_new)


# In[18]:


df_new.head()


# In[19]:


df_new.ILADI.value_counts()


# In[20]:


#Sadece İzmir genelini aldım.
df_new = df_new.query("ILADI == 'İzmir'").reset_index(drop=True)


# In[21]:


df_new.head()


# In[32]:


df_new.MAIN_CHANNEL.unique()


# In[22]:


df_new.info()


# In[25]:


# data = Class'ları belli olan dataframe
data = pd.merge(all_data[["outlet_number","high_traffic"]], df_new, on='outlet_number', how='left')


# In[26]:


# Sadece İzmir içeren toplam 81 satır var. 52-> nonhigh, 29-> high traffic
izmir_data = data.query("ILADI=='İzmir'").reset_index(drop=True)


# In[29]:


izmir_data.head()


# In[31]:


# real values vs preds. bakılırken bu datayı kullanacaksın.
izmir_data.MAIN_CHANNEL.unique()


# ### Data Prep.

# In[33]:


prep_data = df_new.copy()


# In[34]:


prep_data.MAIN_CHANNEL.unique()


# In[35]:


# sadece traditional olanları al.
prep_data = prep_data.query("MAIN_CHANNEL=='TRADITIONAL RETAIL'").reset_index(drop=True)


# In[36]:


model_data = prep_data.copy()


# In[37]:


len(model_data)


# In[38]:


model_data.head()


# In[39]:


model_data["SES"].unique()


# #### Label Encoding

# In[40]:


model_data["SES"] = model_data["SES"].replace({'A': 6, 'B': 5, 'C1': 4, 'C2': 3, 'D': 2, 'E': 1})


# #### İçeceklerin Oranları

# In[41]:


brand_list = ["BURN"
             ,"CAPPY"
             ,"CC_LIGHT"
             ,"CC_NO_SUGER"
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


# In[42]:


m = model_data[brand_list]


# In[44]:


m['total'] = m.sum(axis=1)


# In[31]:


m.head(3)


# In[45]:


model_data["BURN"] = m["BURN"]/m["total"]
model_data["CAPPY"] = m["CAPPY"]/m["total"]
model_data["CC_LIGHT"] = m["CC_LIGHT"]/m["total"]
model_data["CC_NO_SUGER"] = m["CC_NO_SUGER"]/m["total"]
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

# In[46]:


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


# In[47]:


model_data['GECE_NUFUS'].replace({0: 0.00001}, inplace=True)
model_data['GUNDUZ_NUFUS'].replace({0: 0.00001}, inplace=True)


# In[48]:


model_data.head(3)


# In[49]:


g = model_data[['GUNDUZ_NUFUS','GECE_NUFUS']]


# In[50]:


model_data["gündüz_gece_oran"] = g["GUNDUZ_NUFUS"]/g["GECE_NUFUS"]


# In[51]:


model_data.head(3)


# In[52]:


model_data = model_data.fillna(0)


# In[53]:


model_data['SES'] = model_data['SES'].astype(str).astype(int)


# In[54]:


model_data.SES.unique()


# In[55]:


model_data = model_data.drop(["outlet_number"                   
                             ,'GUNDUZ_NUFUS'
                             ,'GECE_NUFUS'],axis=1)


# In[56]:


model_data.info()


# #### Scaling and Normal Dist.

# In[57]:


# define a method to scale data, looping thru the columns, and passing a scaler
def scale_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in data.columns:
        data[col] = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


# In[58]:


def normal_dist(data):
    for col in data.columns:
        data[col] = data[col].apply(lambda x: boxcox1p(x,0.25))
        stats.boxcox(data[col])[0]
    return data


# In[59]:


def log_data(data):
    for col in data.columns:
        data[col] = data[col].apply(lambda x: np.log(x))
    return data


# In[93]:


copy_data = model_data.copy()


# In[94]:


copy_data.head(2)


# In[95]:


copy_data = copy_data.drop(["IL_BAZLI_TURIST_YERLI_ISLETME","IL_BAZLI_TURIST_YBN_ISLETME"],axis=1)


# In[96]:


copy_data[copy_data < 0] = 0


# In[97]:


copy_data.replace({0: 0.00001}, inplace=True)


# #### Normal Dist. Part 

# Comment satırı içinde olan sütunlar, sabit(aynı) değerler içerdiğinden normalize edilemeyen değerlerdir. (ilerde buna göre bir fonksiyon yazabilirsin)

# In[98]:


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
#copy_data["HAVAYOLU"] = stats.boxcox(copy_data["HAVAYOLU"])[0]
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
copy_data["CC_NO_SUGER"] = stats.boxcox(copy_data["CC_NO_SUGER"])[0]
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


# In[99]:


sns.displot(model_data['AYLIK_HARCAMA'])


# In[100]:


sns.displot(copy_data["AYLIK_HARCAMA"])


# In[101]:


sns.displot(model_data["LOKANTAOTEL"])


# In[102]:


sns.displot(copy_data["LOKANTAOTEL"])


# In[103]:


model_data.SES.hist()


# In[104]:


copy_data.SES.hist()


# #### Scaling Part

# In[105]:


scale_df = copy_data.copy()


# In[106]:


scale_df = scale_data(scale_df)


# In[107]:


sns.displot(copy_data['AYLIK_HARCAMA'])


# In[108]:


sns.displot(scale_df["AYLIK_HARCAMA"])


# # Model 

# In[109]:


scale_df.isna().all().all()


# ### K-Means

# In[110]:


km_model = KMeans(n_clusters=2, random_state=42, max_iter=1000000, init='k-means++')
km_model.fit(scale_df)
km_preds = km_model.predict(scale_df)


# In[111]:


final_df = prep_data.copy()


# In[112]:


final_df["k_means_preds"] = km_preds


# In[113]:


final_df


# #### Real Values vs Predictions

# In[115]:


# Real Values
izmir_data.head(3)


# In[117]:


comparison_df = pd.merge(izmir_data, final_df, on='outlet_number')


# In[118]:


conf_df = comparison_df[["high_traffic","k_means_preds"]]
conf_df


# In[121]:


TP = len(conf_df.query("high_traffic == 1 & k_means_preds == 1")) 
FP = len(conf_df.query("high_traffic == 1 & k_means_preds == 0"))
FN = len(conf_df.query("high_traffic == 0 & k_means_preds == 1"))
TN = len(conf_df.query("high_traffic == 0 & k_means_preds == 0"))


# In[122]:


print("TP is:", TP)
print("FP is:", FP)
print("FN is:", FN)
print("TN is:", TN)


# In[171]:


print("Accuracy is: %", (TP+TN)/(TP+TN+FN+FP)*100)


# ## Decision Points for K-Means

# In[176]:


scale_df.head()


# In[173]:


# decision tree için kullanılacak bu data; içinde scale edilmiş değerler yanında hier. clustering prediction'larını da içerir.
km_decision_df = scale_df.copy()


# In[177]:


km_decision_df['k_means_preds'] = km_preds


# In[178]:


from sklearn.model_selection import train_test_split

X = km_decision_df.drop('k_means_preds', axis=1)
Y = km_decision_df['k_means_preds']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=41)
    
km_dc = DecisionTreeClassifier(criterion="entropy", random_state=42) 
km_dc.fit(X_train, Y_train)
km_dc_pred = km_dc.predict(X_test)

km_d_text = tree.export_text(km_dc)
print(km_d_text)


# In[180]:


from sklearn import metrics
metrics.accuracy_score(Y_test, km_dc_pred)


# In[183]:


km_dc.feature_importances_


# In[184]:


pd.Series(km_dc.feature_importances_, index=X_train.columns).nlargest(15).plot(kind='barh')


# ### Hierarchical Clustering

# In[123]:


scale_df.info()


# In[124]:


from sklearn.cluster import AgglomerativeClustering

hrc_model = AgglomerativeClustering(n_clusters=2, linkage='ward', affinity='euclidean')
hrc_preds = hrc_model.fit_predict(scale_df)


# In[125]:


final_df["hrc_preds"] = hrc_preds


# In[126]:


final_df["hrc_preds"].unique()


# In[127]:


final_df['hrc_preds'].value_counts()


# In[128]:


final_df.head()


# In[130]:


comparison_df2 = pd.merge(izmir_data, final_df, on='outlet_number')


# In[131]:


conf_df2 = comparison_df2[["high_traffic","hrc_preds"]]
conf_df2


# In[132]:


TP_ = len(conf_df2.query("high_traffic == 1 & hrc_preds == 1")) 
FP_ = len(conf_df2.query("high_traffic == 1 & hrc_preds == 0"))
FN_ = len(conf_df2.query("high_traffic == 0 & hrc_preds == 1"))
TN_ = len(conf_df2.query("high_traffic == 0 & hrc_preds == 0"))


# In[133]:


print("TP is:", TP_)
print("FP is:", FP_)
print("FN is:", FN_)
print("TN is:", TN_)


# In[172]:


print("Accuracy is: %", (TP_+TN_)/(TP_+TN_+FN_+FP_)*100)


# <img src="image2.png" width=350 height=350 />

# In[136]:


# extract coordinates
final_df["x"] = final_df["GEOGPOINT"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[1])
final_df["y"] = final_df["GEOGPOINT"].apply(lambda x: x.split(")")[0].split("POINT(")[1].split(" ")[0])


# In[137]:


final_df["y"] = pd.to_numeric(final_df["y"], downcast="float")
final_df["x"] = pd.to_numeric(final_df["x"], downcast="float")


# ## Decision Points

# In[138]:


# decision tree için kullanılacak bu data; içinde scale edilmiş değerler yanında hier. clustering prediction'larını da içerir.
decision_df = scale_df.copy()


# In[139]:


decision_df['hrc_preds'] = hrc_preds


# In[141]:


from sklearn.model_selection import train_test_split

x = decision_df.drop('hrc_preds', axis=1)
y = decision_df['hrc_preds']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=41)
    
dc = DecisionTreeClassifier(criterion="entropy", random_state=42) 
dc.fit(x_train, y_train)
dc_pred = dc.predict(x_test)

d_text = tree.export_text(dc)
print(d_text)


# In[142]:


from sklearn import metrics
metrics.accuracy_score(y_test, dc_pred)


# In[143]:


dc.feature_importances_


# In[144]:


pd.Series(dc.feature_importances_, index=x_train.columns).nlargest(15).plot(kind='barh')


# In[145]:


analysis_df = final_df.drop([ 
"MAIN_CHANNEL"                                             
,"YAS_CLUSTER"                    
,"ILADI"                          
,"ILCEADI"  
,"GEOGPOINT"
,"IDARIID"
,"HANE_BUYUKLUGU"
,"k_means_preds"
,"x","y"
],axis=1)


# In[146]:


final_high = analysis_df.query("hrc_preds == 1")
final_low = analysis_df.query("hrc_preds == 0") 


# In[147]:


final_high.describe()


# In[148]:


final_low.describe()


# #### High Traffic

# In[149]:


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

# In[150]:


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


# ### Graph

# In[151]:


column_names = ["BURN"
,"CAPPY"                            
,"CC_LIGHT"                         
,"CC_NO_SUGER"                      
,"COCACOLA"                         
,"COCACOLA_ENERGY"                  
,"DAMLA_MINERA"                     
,"DAMLA_WATER"                      
,"EXOTIC"                           
,"FANTA"                            
,"FUSETEA"                          
,"MONSTER"                          
,"POWERADE"                         
,"SCHWEPPES"]                      


# In[152]:


final_high_brands = final_high[column_names].reset_index(drop=True)


# In[153]:


high_brands_sum = pd.DataFrame(final_high_brands.sum()).reset_index()


# In[154]:


high_brands_sum.rename({"index": "brand", 0: "values",} ,axis=1,inplace=True)


# In[155]:


high_brands_sum


# In[156]:


fig = px.pie(high_brands_sum, values='values', names="brand", title='Brand Dist. in High Traffic Cluster')
fig.show()


# In[157]:


final_high.groupby('SES')['outlet_number'].sum().sort_values()


# In[158]:


non_high_brands = final_low[column_names].reset_index(drop=True)


# In[159]:


non_brands_sum = pd.DataFrame(non_high_brands.sum()).reset_index()


# In[160]:


non_brands_sum.rename({"index": "brand", 0: "values",} ,axis=1,inplace=True)


# In[161]:


non_brands_sum.query("brand == 'CC_LIGHT'")


# In[162]:


fig = px.pie(non_brands_sum, values='values', names="brand", title='Brand Dist. in Non-high Traffic Cluster')
fig.show()


# In[163]:


final_low.groupby('SES')['outlet_number'].sum().sort_values()


# ----

# ## Score Based

# In[164]:


high_df = ist_data.query("high_traffic==1")
non_df = ist_data.query("high_traffic==0")


# In[165]:


high_df.describe()


# In[166]:


non_df.describe()


# In[167]:


stophere


# ----

# In[170]:


# bigquery'e tabloyu ekleme
# Database Connection
from google.cloud import bigquery, bigquery_storage_v1beta1

client = bigquery.Client()

table_id = "coca-cola-data-lake.predictive_order.high_traffic_clusters_izmir" 

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("outlet_name", bigquery.enums.SqlTypeNames.STRING),
    ],  write_disposition="WRITE_TRUNCATE"
)

job = client.load_table_from_dataframe(final_df, table_id, job_config=job_config)  
job.result()  

table = client.get_table(table_id)  
print("Loaded {} rows and {} columns to {}".format(table.num_rows, len(table.schema), table_id))


# #### Save as CSV

# In[186]:


csv_df = final_df.copy()


# In[187]:


csv_df = csv_df.rename({"hrc_preds": "high_traffic"}, axis=1)
csv_df = csv_df.drop("k_means_preds", axis=1)


# In[189]:


csv_df.to_csv("izmir_high_traffic.csv")

