
# coding: utf-8

# # Estimated Delivery Date: A case study
# 
# When shopping at Farfetch, customers are offered a catalog of thousands of products that are sourced from different partners spread around the world. During the shopping experience, customers must have an accurate estimate of their purchase delivery date to manage their expectations.
# 
# The goal of this case study is to **understand the Expected Delivery Date of an order and provide a robust and accurate delivery date estimate** to the customer. To support your understanding of the problem and development of the challenge you will receive a dataset split in training and test set. Further details are given in the Data Instructions attached to the case.

# ### Data Ingestion
# 
# We are going to start this study by ingesting the given dataset and verifying its data quality, in order to manage any problem that might appear.

# In[290]:


import matplotlib
import time
import numpy as np
import pandas as pd
import pandas_profiling as pp

from geopy import distance
from geopy import geocoders 
from geopy.geocoders import Nominatim
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:


get_ipython().magic('matplotlib inline')


# In[3]:


dataFile = pd.read_csv('data/train.csv')
df= pd.DataFrame(dataFile)


# In[4]:


# First data inspection
df.head()


# In[5]:


# Checking out dataset shape
df.shape


# In[6]:


# Data types and info
df.info()


# In[7]:


# Verifying duplicated data
dfDuplicated = pd.DataFrame.duplicated(df,keep='first')
dfDuplicated.sum()


# ### Exploratory Data Analysis
# 
# Let us investigate the dataset more deeply.

# In[8]:


# Detailed report
pr = pp.ProfileReport(df)


# In[9]:


pr


# In[20]:


# Analyzing missing Partner information
df[df.PartnerLatitude.isnull()]


# In[23]:


# Analyzing missing Partner information
df[(df.PartnerCountry.isnull()) & (df.PartnerLatitude.isnull())]


# In[25]:


# Analyzing missing Customer information
df[df.CustomerLatitude.isnull()]


# In[26]:


# Analyzing missing Customer information
df[(df.CustomerCountry.isnull()) & (df.CustomerLatitude.isnull())]


# In[10]:


# Numerical statistics
df.describe()


# In[17]:


# Delivery Time Histogram
df.DeliveryTime.plot.hist(bins=50)


# ### Feature Engineering
# 
# In this section we are going to explore the data and create the features for the model.
# 
# Previous analysis shows that:
# - Some features can be discarded because they have the same meaning, e.g.: (Lat+Long) or (Country+City).
# - ID information doesn't seem relevant for the current problem (OrderID, PartnerID, etc).
# - IsHazMat is a constant in this dataset. If we keep it, the model won't know what to do with a different information, so it can be discarded in this project (altough it raises a concern on having different cases for this feature in a new dataset).
# - Categories can (and will) be encoded, keeping the most frequent ones, while the least frequent categories are going to be blended in a category called "others".
# - Sub-categories have a high cardinallity, so they are not suitable for efficient one-hot encoding and will be dropped
# - Some important information are missing (e.g.: Lat, Long). We will try to replace it whenever we can, otherwise it is going to be discarded.
# - We will create a new feature by calculating the distance between partner and customer using their lat/long info, because it seems like a very important feature.

# In[255]:


# Creates a new dataset where data is going to be transformed (mf=model features)
mf = df.copy()


# #### Dropping unused

# In[256]:


# Dropping features we are not going to use
def dropUnused(modelFeatures):
    modelFeatures.drop(labels='IsHazmat', axis=1, inplace=True)
    modelFeatures.drop(labels='OrderLineID', axis=1, inplace=True)
    modelFeatures.drop(labels='OrderCodeID', axis=1, inplace=True)
    modelFeatures.drop(labels='PartnerID', axis=1, inplace=True)
    modelFeatures.drop(labels='TariffCode', axis=1, inplace=True)
    modelFeatures.drop(labels='CustomerCountry', axis=1, inplace=True)
    modelFeatures.drop(labels='CustomerCity', axis=1, inplace=True)
    modelFeatures.drop(labels='PartnerCountry', axis=1, inplace=True)
    modelFeatures.drop(labels='PartnerCity', axis=1, inplace=True)
    modelFeatures.drop(labels='DdpSubcategory', axis=1, inplace=True)    
    modelFeatures.drop(labels='Category2ndLevel', axis=1, inplace=True)        
    return modelFeatures


# In[259]:


# Applying changes to dataset
mf = dropUnused(mf)


# In[260]:


# Verifying function
mf.head()


# #### Trying to figure out coordenates from City/Coutry
# 
# Altough technically possible on a few examples, the webservice doesn't allow us to iterate through so many rows, so it was not possible to do this transformation.

# In[257]:


# Filling in missing Lat/Long - Webservice did not support multiple calls
def defLatLong(modelFeatures):
    geolocator = Nominatim(user_agent="defLatLong")
    
    for index, row in modelFeatures.iterrows():
        time.sleep(5)
        if pd.isnull(row['CustomerLongitude']):
            location = geolocator.geocode("{},{}".format(row['CustomerCity'],row['CustomerCountry']))
            if pd.notnull(location):
                row['CustomerLongitude'] = location.longitude
                row['CustomerLatitude'] = location.latitude
            
        if pd.isnull(row['PartnerLongitude']):
            location = geolocator.geocode("{},{}".format(row['PartnerCity'],row['PartnerCountry']))
            if pd.notnull(location):
                row['PartnerLongitude'] = location.longitude
                row['PartnerLatitude'] = location.latitude
        
    return modelFeatures


# #### Dropping Nulls

# In[258]:


# Drop Nulls
def dropNulls(modelFeatures):
    modelFeatures.dropna(subset=['CustomerLongitude','CustomerLatitude','PartnerLongitude','PartnerLatitude'], how='any', axis=0, inplace=True)    
    return modelFeatures


# In[261]:


# Verifying function
mf.shape


# In[262]:


# Dropping nulls
mf = dropNulls(mf)


# In[263]:


# Verifying function
mf.shape


# In[264]:


# Verifying function
mf.PartnerLongitude.isnull().sum()


# #### Calculating distance between partner and customer

# In[265]:


# Calculates distance between coordinates
def calcDist(modelFeatures):       
    modelFeatures['DistanceKM'] = modelFeatures.apply(lambda x: distance.distance( (x['CustomerLatitude'],x['CustomerLongitude']) , (x['PartnerLatitude'],x['PartnerLongitude']) ).km, axis=1)
    return modelFeatures


# In[266]:


mf = calcDist(mf)


# In[267]:


# Verifying function
mf.plot.scatter(x='DistanceKM',y='DeliveryTime')


# In[268]:


# Distance Histogram
mf.DistanceKM.plot.hist(bins=50)


# #### Preparing categorical variables

# In[271]:


# Preparing categorical variables for one hot encoding
def prepareEncoding(modelFeatures):
    modelFeatures['DeliveryType'] = modelFeatures.DeliveryType.apply(str).apply(lambda x: 1 if x == 'Express' else 0)
    modelFeatures['DdpCategory'] = modelFeatures.DdpCategory.apply(str).apply(lambda x: 'CategoryClothing' if x == 'Clothing & Accessories' else ('CategoryFootwear' if x[:8] == 'Footwear' else 'CategoryOthers'))
    modelFeatures['Category1stLevel'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 'LevelClothing' if x == 'Clothing' else ('LevelShoes' if x == 'Shoes'  else ('LevelBags' if x == 'Bags'  else 'LevelOthers')))
    return modelFeatures


# In[272]:


# Applying changes
mf = prepareEncoding(mf)


# In[273]:


# Verifying changes
mf.head()


# #### Getting month from data

# In[278]:


# Get month from date due to seasonality
def getMonth(modelFeatures):
    modelFeatures['OrderDate'] = modelFeatures.OrderDate.str.slice(5,7)
    return modelFeatures


# In[279]:


# Applying changes
mf = getMonth(mf)
mf.rename(index=str, columns={'OrderDate':'OrderMonth'}, inplace=True)


# In[282]:


# Verifying changes
mf.head()


# #### Encoding categorical variables

# In[287]:


def encodeCategories(modelFeatures):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    
    modelFeatures['DdpCategoryNum'] = label_encoder.fit_transform(modelFeatures['DdpCategory'])
    modelFeatures['Category1stLevelNum'] = label_encoder.fit_transform(modelFeatures['Category1stLevel'])
    
    oheDdp = onehot_encoder.fit_transform(modelFeatures['DdpCategoryNum'])
    
    return modelFeatures


# In[288]:


# Applying changes
mf = encodeCategories(mf)


# In[289]:


# Verifying changes
mf.head()

