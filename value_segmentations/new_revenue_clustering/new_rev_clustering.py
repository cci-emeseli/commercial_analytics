#!/usr/bin/env python
# coding: utf-8

# <h1><center>Revenue Segmentation</center></h1>

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
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, Birch, MeanShift, AgglomerativeClustering
from scipy.special import boxcox1p
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

sns.set_style("whitegrid")
# sns.color_palette('bright')
sns.set_palette('bright')


# In[2]:


data = pd.read_excel('../new_rev_clustering/revenue_geleneksel.xlsx')


# In[3]:


data.head(3)


# In[4]:


df = data[["outlet_number", "Rounded Sparc", "Rounded FIT"]]


# In[5]:


df.rename(columns = {'Rounded Sparc':'sparc', 'Rounded FIT':'fit'}, inplace = True)


# In[6]:


df.head()


# In[7]:


len(df)


# In[8]:


df.outlet_number.nunique()


# In[9]:


df.isnull().sum()


# In[10]:


df.replace(np.nan, 0, inplace=True)


# In[11]:


df.isnull().sum()


# <h2><center>Value Segmentation with Sparc</center></h2>

# In[12]:


sparc_df = df[["outlet_number", "sparc"]]


# In[16]:


## filter 1000+ revenue sales.
sparc_df = sparc_df[sparc_df['sparc']>1000] 


# In[17]:


## filter 500.000- revenue sales.
sparc_df = sparc_df[sparc_df['sparc']<500000] 


# In[18]:


sparc_df.isnull().sum()


# In[19]:


len(sparc_df)


# In[20]:


plt.rcParams["figure.figsize"] = (10,8)
plt.plot(sparc_df.sparc)


# #### Normalization

# In[21]:


# Normalize data
def normal_dist(data):
    return (data.apply(lambda x: boxcox1p(x,0.25)))


# In[22]:


sparc_df["sparc"].replace(0, 0.001, inplace=True)


# In[23]:


sparc_df["sparc_norm"] = normal_dist(sparc_df["sparc"])


# In[24]:


# filtering negative amount values (we have *42* negative values here)
sparc_df[sparc_df['sparc_norm']<0] = 0


# In[25]:


model_data = sparc_df['sparc_norm']


# In[26]:


model_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[27]:


model_data.replace(np.nan, 0, inplace=True)


# In[28]:


model_data.describe()


# In[29]:


sns.displot(model_data)


# #### Model

# In[30]:


## KMeans Model
sparc_model = KMeans(n_clusters=4, random_state=42, init='k-means++')
sparc_model.fit(model_data.values.reshape(-1, 1))
sparc_preds = sparc_model.predict(model_data.values.reshape(-1, 1))
sparc_df["cluster"] = sparc_preds


# In[31]:


sparc_df["cluster"].value_counts()


# In[32]:


sparc_df.head()


# In[33]:


sns.scatterplot(data=sparc_df, hue='cluster', y='sparc', x='sparc_norm', palette=['green','red','dodgerblue',"orange"])
plt.title('Cluster Dist.')
plt.show()


# In[34]:


sparc_df['cluster'] = sparc_df['cluster'].apply(lambda x: 
                                                0 if x==2 # red
                                                else 1 if x==1 # orange
                                                else 2 if x==3 # green
                                                else 3 # blue
                                                )


# In[35]:


sparc_df['cluster'].value_counts()


# In[36]:


pie=sparc_df.groupby('cluster').size().reset_index()
pie.columns=['cluster','value']
px.pie(pie,values='value',names='cluster', title='Cluster Dist.')


# ### Decision Points

# In[37]:


dc_sparc = DecisionTreeClassifier(random_state=42)
dc_sparc.fit(sparc_df['sparc'].values.reshape(-1,1), sparc_df['cluster'])
dc_sp_preds = dc_sparc.predict(sparc_df['sparc'].values.reshape(-1,1))
clf_text = tree.export_text(dc_sparc)
print(clf_text)


# <h2><center>Value Segmentation with Fit Data</center></h2>

# In[38]:


fit_df = df[["outlet_number", "fit"]]


# In[39]:


## filter 1000+ revenue sales.
fit_df = fit_df[fit_df['fit']>1000] 


# In[40]:


## filter 500000- revenue sales.
fit_df = fit_df[fit_df['fit']<500000] 


# In[41]:


fit_df.isnull().sum()


# In[42]:


plt.rcParams["figure.figsize"] = (10,8)
plt.plot(fit_df.fit)


# In[43]:


len(fit_df[fit_df["fit"]<0]) # 0 negative value


# In[44]:


len(fit_df[fit_df["fit"]==0]) # 0 zero value


# #### Normalization

# In[45]:


# Normalize data
def normal_dist(data):
    return (data.apply(lambda x: boxcox1p(x,0.25)))


# In[46]:


fit_df["fit"].replace(0, 0.001, inplace=True)


# In[47]:


fit_df["fit_norm"] = normal_dist(fit_df["fit"])


# In[48]:


# filtering negative amount values (we have *42* negative values here)
fit_df[fit_df["fit_norm"]<0] = 0


# In[49]:


model_data2 = fit_df['fit_norm']


# In[50]:


model_data2.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[51]:


model_data2.replace(np.nan, 0, inplace=True)


# In[52]:


len(model_data2) # before and after -> 178989 


# In[53]:


model_data2.describe()


# In[54]:


sns.displot(model_data2)


# #### Model

# In[55]:


## KMeans Model
fit_model = KMeans(n_clusters=4, random_state=42, init='k-means++')
fit_model.fit(model_data2.values.reshape(-1, 1))
fit_preds = fit_model.predict(model_data2.values.reshape(-1, 1))
fit_df["cluster"] = fit_preds


# In[56]:


fit_df["cluster"].value_counts()


# In[57]:


sns.scatterplot(data=fit_df, hue='cluster', y='fit', x='fit_norm', palette=['green','red','dodgerblue',"orange"])
plt.title('Cluster Dist.')
plt.show()


# In[58]:


fit_df['cluster'] = fit_df['cluster'].apply(lambda x: 
                                                0 if x==2
                                                else 1 if x==0
                                                else 2 if x==3
                                                else 3 
                                                )


# In[59]:


fit_df['cluster'].value_counts()


# In[60]:


pie=fit_df.groupby('cluster').size().reset_index()
pie.columns=['cluster','value']
px.pie(pie,values='value',names='cluster', title='Cluster Dist.')


# ### Decision Points

# In[61]:


dc_fit = DecisionTreeClassifier(random_state=42)
dc_fit.fit(fit_df['fit'].values.reshape(-1,1), fit_df['cluster'])
dc_fit_preds = dc_fit.predict(fit_df['fit'].values.reshape(-1,1))
clf_text = tree.export_text(dc_fit)
print(clf_text)


# ## Results 

# In[62]:


fit_results = fit_df[["outlet_number","fit","cluster"]].reset_index(drop=True)


# In[63]:


fit_results.head()


# In[64]:


sparc_results = sparc_df[["outlet_number","sparc","cluster"]].reset_index(drop=True)


# In[65]:


sparc_results.head()


# In[68]:


durr


# In[66]:


fit_results.to_csv("fit_results.csv")


# In[67]:


sparc_results.to_csv("sparc_results.csv")


# ----

# In[14]:


spark = df[["outlet_number", "sparc"]]


# In[15]:


## filter 1000+ revenue sales.
len(spark[spark['sparc']<1000])


# In[72]:


## filter 1000- and 5000000+ revenue sales.
sp1 = spark[spark['sparc']<1000]
sp2 = spark[spark['sparc']>500000]

frames = [sp1, sp2]
sp_result = pd.concat(frames)
sp_result = sp_result.reset_index(drop=True)


# In[80]:


sp_result.to_csv("sparc_outliers.csv")


# In[85]:


len(sp_result)


# In[83]:


fid = df[["outlet_number", "fit"]]

## filter 1000- and 5000000+ revenue sales.
f1 = fid[fid['fit']<1000]
f2 = fid[fid['fit']>500000]

frames2 = [f1, f2]
fit_result = pd.concat(frames2)
fit_result = fit_result.reset_index(drop=True)


# In[86]:


len(fit_result)


# In[88]:


fit_result


# In[87]:


fit_result.to_csv("fit_outliers.csv")


# ----
