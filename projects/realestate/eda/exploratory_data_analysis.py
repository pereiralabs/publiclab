
# coding: utf-8

# In[1]:


#Libraries
import glob
import json
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


# # Real Estate Data
# 
# We will analyze real estate data from a real estate website.

# ## Data Ingestion

# In[10]:


#Reading file
newFile = '/home/DAITAN/ftsantos/workingdir/pro/bitbucket/workingdir/per/git/lab/projects/realestate/data/immobilienscout24_berlin_20190113.json'

#Parsing Json
dfNew = pd.read_json(path_or_buf=newFile, typ='frame', orient='values', dtype=False, convert_dates=False, lines=True)        


# ## Analyzing the dataset
# 
# In this section there are just some small analysis to understand the dataset.

# In[13]:


dfNew.shape


# In[16]:


#dfNew.describe()


# In[15]:


dfNew.columns


# In[17]:


dfNew.dtypes


# In[18]:


dfNew.head()


# ## Data Prep
# 
# The data needs to be prepared in order to be used for further analysis.

# In[19]:


#Expands the Data field
dfData = dfNew.data.apply(pd.Series)


# In[20]:


dfData.head()


# In[37]:


#Seeing all columns
aux = 0
for col in dfData.columns:
    print (dfData.columns[aux])
    aux = aux + 1


# In[44]:


dfData.dtypes

