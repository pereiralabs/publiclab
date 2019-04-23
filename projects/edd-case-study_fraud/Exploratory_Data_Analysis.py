
# coding: utf-8

# # Estimated Delivery Date: A case study
# 
# When shopping at Farfetch, customers are offered a catalog of thousands of products that are sourced from different partners spread around the world. During the shopping experience, customers must have an accurate estimate of their purchase delivery date to manage their expectations.
# 
# The goal of this case study is to **understand the Expected Delivery Date of an order and provide a robust and accurate delivery date estimate** to the customer. To support your understanding of the problem and development of the challenge you will receive a dataset split in training and test set. Further details are given in the Data Instructions attached to the case.

# ### Data Ingestion
# 
# We are going to start this study by ingesting the given dataset and verifying its data quality, in order to manage any problem that might appear.

# In[59]:


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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


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


# In[10]:


# Analyzing missing Partner information
df[df.PartnerLatitude.isnull()]


# In[11]:


# Analyzing missing Partner information
df[(df.PartnerCountry.isnull()) & (df.PartnerLatitude.isnull())]


# In[12]:


# Analyzing missing Customer information
df[df.CustomerLatitude.isnull()]


# In[13]:


# Analyzing missing Customer information
df[(df.CustomerCountry.isnull()) & (df.CustomerLatitude.isnull())]


# In[14]:


# Numerical statistics
df.describe()


# In[15]:


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

# In[16]:


# Creates a new dataset where data is going to be transformed (mf=model features)
mf = df.copy()


# #### Dropping unused

# In[17]:


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


# In[18]:


# Applying changes to dataset
mf = dropUnused(mf)


# In[19]:


# Verifying function
mf.head()


# #### Trying to figure out coordenates from City/Coutry
# 
# Altough technically possible on a few examples, the webservice doesn't allow us to iterate through so many rows, so it was not possible to do this transformation.

# In[20]:


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

# In[21]:


# Drop Nulls
def dropNulls(modelFeatures):
    modelFeatures.dropna(subset=['CustomerLongitude','CustomerLatitude','PartnerLongitude','PartnerLatitude'], how='any', axis=0, inplace=True)    
    return modelFeatures


# In[22]:


# Verifying function
mf.shape


# In[23]:


# Dropping nulls
mf = dropNulls(mf)


# In[24]:


# Verifying function
mf.shape


# In[25]:


# Verifying function
mf.PartnerLongitude.isnull().sum()


# #### Calculating distance between partner and customer

# In[26]:


# Calculates distance between coordinates
def calcDist(modelFeatures):       
    modelFeatures['DistanceKM'] = modelFeatures.apply(lambda x: distance.distance( (x['CustomerLatitude'],x['CustomerLongitude']) , (x['PartnerLatitude'],x['PartnerLongitude']) ).km, axis=1)
    return modelFeatures


# In[27]:


mf = calcDist(mf)


# In[28]:


# Verifying function
mf.plot.scatter(x='DistanceKM',y='DeliveryTime')


# In[29]:


# Distance Histogram
mf.DistanceKM.plot.hist(bins=50)


# #### Preparing categorical variables

# In[30]:


# Preparing categorical variables for one hot encoding
def prepareEncoding(modelFeatures):
    modelFeatures['DeliveryType'] = modelFeatures.DeliveryType.apply(str).apply(lambda x: 1 if x == 'Express' else 0)
    modelFeatures['DdpCategory'] = modelFeatures.DdpCategory.apply(str).apply(lambda x: 'CategoryClothing' if x == 'Clothing & Accessories' else ('CategoryFootwear' if x[:8] == 'Footwear' else 'CategoryOthers'))
    modelFeatures['Category1stLevel'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 'LevelClothing' if x == 'Clothing' else ('LevelShoes' if x == 'Shoes'  else ('LevelBags' if x == 'Bags'  else 'LevelOthers')))
    return modelFeatures


# In[31]:


# Applying changes
mf = prepareEncoding(mf)


# In[32]:


# Verifying changes
mf.head()


# #### Getting month from data

# In[33]:


# Get month from date due to seasonality
def getMonth(modelFeatures):
    modelFeatures['OrderDate'] = modelFeatures.OrderDate.str.slice(5,7)
    modelFeatures.rename(index=str, columns={'OrderDate':'OrderMonth'}, inplace=True)
    return modelFeatures


# In[34]:


# Applying changes
mf = getMonth(mf)


# In[35]:


# Verifying changes
mf.head()


# #### Encoding categorical variables

# In[36]:


def encodeCategories(modelFeatures):
    modelFeatures['CategoryClothing'] = modelFeatures.DdpCategory.apply(str).apply(lambda x: 1 if x == 'CategoryClothing' else 0)
    modelFeatures['CategoryFootwear'] = modelFeatures.DdpCategory.apply(str).apply(lambda x: 1 if x == 'CategoryFootwear' else 0)
    modelFeatures['CategoryOthers'] = modelFeatures.DdpCategory.apply(str).apply(lambda x: 1 if x == 'CategoryOthers' else 0)

    modelFeatures['Category1stLevel'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 'LevelClothing' if x == 'Clothing' else ('LevelShoes' if x == 'Shoes'  else ('LevelBags' if x == 'Bags'  else 'LevelOthers')))
    modelFeatures['LevelClothing'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 1 if x == 'LevelClothing' else 0)
    modelFeatures['LevelShoes'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 1 if x == 'LevelShoes' else 0)
    modelFeatures['LevelBags'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 1 if x == 'LevelBags' else 0)
    modelFeatures['LevelOthers'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 1 if x == 'LevelOthers' else 0)
    
    return modelFeatures


# In[37]:


# Applying changes
mf = encodeCategories(mf)


# In[38]:


# Verifying changes
mf.head()


# #### Dropping already used features

# In[39]:


# Dropping features we are not going to use
def dropUsed(modelFeatures):
    modelFeatures.drop(labels='DdpCategory', axis=1, inplace=True)
    modelFeatures.drop(labels='Category1stLevel', axis=1, inplace=True)
    
    return modelFeatures


# In[40]:


# Applying changes
mf = dropUsed(mf)


# In[41]:


# Verifying changes
mf.head()


# ### Model Selection
# 
# In this section we are going to use the features that we have prepared, in order to choose the best model.
# 
# We will train the following models:
# - Linear Regression
# - Decision Tree
# - Random Forest
# - AdaBoost

# In[50]:


# Creating the dataframe
modelDf = df.copy()


# In[51]:


# Applying feature engineering transformations
modelDf = dropUnused(modelDf)
modelDf = dropNulls(modelDf)
modelDf = calcDist(modelDf)
modelDf = prepareEncoding(modelDf)
modelDf = getMonth(modelDf)
modelDf = encodeCategories(modelDf)
modelDf = dropUsed(modelDf)


# In[52]:


# Verifying changes
modelDf.head()


# In[56]:


# Splitting X and Y
X = modelDf.drop(['DeliveryTime'], axis=1)
y = modelDf['DeliveryTime']

#Scaling X
col_names = ['CustomerLatitude', 'CustomerLongitude','PartnerLatitude','PartnerLongitude','DistanceKM']
scaler = StandardScaler().fit(X[col_names])
X[col_names] = scaler.transform(X[col_names])
X.head()


# In[57]:


# Splitting train and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[82]:


#Definig hyperparameters space
params ={
    'LinearRegression':{'fit_intercept':[True,False]},
    'DecisionTree':{'criterion':['mse','friedman_mse','mae'], 
                    'splitter':['best','random']},
    'RandomForest':{'n_estimators':[10,50,100], 
                    'criterion':['mse','mae']},
    'AdaBoost':{'base_estimator':[DecisionTreeRegressor(max_depth=3),
                    DecisionTreeRegressor(max_depth=5),
                    DecisionTreeRegressor(max_depth=10)],
                'n_estimators':[50,75,100],
                'learning_rate':[0.1,1,10],
                'loss':['linear','square','exponential']}
}


# In[83]:


#Defining models
models = {
    'LinearRegression':LinearRegression(),
    'DecisionTree':DecisionTreeRegressor(),
    'RandomForest':RandomForestRegressor(),
    'AdaBoost':AdaBoostRegressor()
}


# In[78]:


#Defining GridSearch selection helper
class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]


# In[84]:


#Executing GridSearch
helper = EstimatorSelectionHelper(models, params)
helper.fit(X_train, y_train, scoring='neg_mean_squared_error', n_jobs=3)


# In[93]:


#Verifying results
helper.score_summary(sort_by='max_score')


# In[97]:


res = helper.score_summary(sort_by='max_score')


# In[101]:


res.iloc[0]['base_estimator']


# In[102]:


res.iloc[0]


# ### Training the best model
# 
# We are now going to train the best model, based on our previous search.

# In[104]:


#Defining model
bestModel = AdaBoostRegressor(
    DecisionTreeRegressor(criterion='mse', max_depth=10, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best'),
    learning_rate = 0.1,
    n_estimators = 75
)


# In[105]:


#Training model
bestModel.fit(X_train,y_train)


# In[106]:


#Predicting
y_pred = bestModel.predict(X_test)


# In[107]:


#Verifying results
resultDf = X_test.copy()
resultDf['DeliveryTime'] = y_test
resultDf['PredictedTime'] = y_pred


# In[110]:


#Verifyng df
resultDf.head(30)

