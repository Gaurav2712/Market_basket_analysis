#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Objective:: Here we are segmenting the customers of a wholesale distributor as per there annual spending on 
diverse product categories like milk,grocery, region etc.
"""


# In[18]:


# reading the file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cust_data= pd.read_csv("C:\\Users\\hp\\Downloads\\cust_data.csv")
cust_data


# In[9]:


# Shape of Data 
cust_data.shape


# In[10]:


# column names
cust_data.columns


# In[7]:


# checking missing values 
cust_data.info()
# and found no missing values


# In[11]:


# checking whats there in channel column:
cust_data['Channel'].value_counts()
# here we have found it is either 1 or 2 
#this refers to the mode of payment 
#  1: online        2: offline


# In[12]:


# checking whats there in region column:
cust_data['Region'].value_counts()
# here are only three distinct values 
# this can be referred as :
# 1: Delhi    2: Noida    3: Gurugram


# In[13]:


# checking statistics of  whole data to check for outliers 
x=cust_data.describe()
pd.DataFrame(x)
x
# other than region and Channel rest are the money spent on different food items
# we will be creating clusters depending on different variables 




# In[ ]:


# data preparation for clustering 
# on analyzing i found that there is variation in the data like
# variables like region and channel have lower magnitude where as
# fresh,milk,groceries,frozen and other have high magnitude
# so to apply K means clustering this difference in magnitude will create problem 
# as K- means is a distance based algorithm
# so let's first bring all variables to same magnitude
# I will create a scaled data ie "data_scaled" for which  clusters will be created 


# In[17]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled= scaler.fit_transform(cust_data)
data_scaled
y= pd.DataFrame(data_scaled)
y


# In[22]:


# Now as the data is scaled so KMeans algorithim can be applied
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=2, init='k-means++')
# Fitting kmeans function on scaled data
y=kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
pred=pd.DataFrame(pred)
pred.columns = ['pred_cluster']
pred


# In[23]:


new_ds= pd.concat([cust_data, pred], axis=1)
new_ds


# In[24]:


new_ds['pred_cluster'].value_counts()
# here machine has created only two clusters which shows 305 persons falling in one category and 135 in the one category


# In[25]:


# here only two clusters were created as instructed to make 
# but the point here is that how many optimal clusters should be made 
# we will see that using elbow curve
# fitting multiple k-means algorithms and storing the values in an empty list
# SSE is sum of squared values
SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)


# In[26]:


# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[28]:


# Looking at the above elbow curve, we can choose any number of clusters between 5 to 8. 
# Let’s set the number of clusters as 5 and fit the model:
# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
pred


# In[29]:


#let’s look at the value count of points in each of the above-formed clusters:

frame = pd.DataFrame(pred)
frame.columns = ['cluster_no']
new_ds= pd.concat([cust_data, frame], axis=1)
new_ds


# In[30]:


# checking the clusters
new_ds['cluster_no'].value_counts()


# In[31]:


#profiling clusters
# exploring cluster zero
df1 = new_ds.query('(cluster_no == 0)')
df1['Channel'].value_counts()
# cluster 0 mostly contains online payers


# In[32]:


df1['Region'].value_counts()
# maximum of customers are from Gurugram region


# In[34]:


x=df1.describe()
profiling_df1 = pd.DataFrame(x)
x
# statistics of online paying customers from Gurugram see the median value 50%


# In[35]:


# Analysing cluster 1:
df1 = new_ds.query('(cluster_no == 1)')
df1['Channel'].value_counts()
# all are offline payers


# In[36]:



df1['Region'].value_counts()
# All are from Gurugram


# In[38]:


#Statistics of all the offline paying customers from Gurugram
x=df1.describe()
profiling_df1 = pd.DataFrame(x)
x


# In[39]:


# Analysing cluster 2:
df1 = new_ds.query('(cluster_no == 2)')
df1['Channel'].value_counts()
# all are the online payers


# In[40]:


df1['Region'].value_counts()
# all are from Gurugram


# In[42]:


x=df1.describe()
profiling_df1 = pd.DataFrame(x)
 # stats of all the online payers from gurugram    
x


# In[45]:


# Analysing cluster 3:
df1 = new_ds.query('(cluster_no == 3 )')
df1['Channel'].value_counts()
# maximum are online payers


# In[46]:


df1['Region'].value_counts()
# People are from delhi and Noida and maximum from delhi


# In[47]:


x=df1.describe()
profiling_df1 = pd.DataFrame(x)
 # stats of all the online payers from Delhi and Noida   
x


# In[48]:


# Analysing cluster 4:
df1 = new_ds.query('(cluster_no == 4 )')
df1['Channel'].value_counts()
# All are the offline payers


# In[50]:


df1['Region'].value_counts()
# maximum from Gurugram

df1['Region'].value_counts()
# maximum from Gu
# In[51]:


x=df1.describe()
profiling_df1 = pd.DataFrame(x)
 # stats of all the online payers from Gurugram  
x


# In[ ]:


# INSIGHTS
# Store is doing good in gurugram online as well as offline 
# we have a very low customer base in noida
# delhi have maximum online payers only
# people are purchasing Fresh items  by online mode only
# people prefer to buy milk by offline mode only
# customer from clusters 0 &  4 are mostly business customers


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




