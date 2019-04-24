
# coding: utf-8

# # Estimated Delivery Date: A case study
# 
# The goal of this case study is to **understand the Expected Delivery Date of an order and provide a robust and accurate delivery date estimate** to the customer. To support your understanding of the problem and development of the challenge you will receive a dataset split in training and test set. Further details are given in the Data Instructions attached to the case.

# ## Data Ingestion
# 
# We are going to start this study by ingesting the given dataset and verifying its data quality, in order to manage any problem that might appear.

# In[1]:


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
from sklearn.ensemble import GradientBoostingRegressor

from joblib import dump, load


# In[2]:


get_ipython().magic('matplotlib inline')


# In[3]:


dataFile = pd.read_csv('data/train.csv')
df = pd.DataFrame(dataFile)


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


# ## Exploratory Data Analysis
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


# ## Feature Engineering
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


# ### Dropping unused

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


# ### Trying to figure out coordenates from City/Coutry
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


# ### Dropping Nulls
# 
# Used in model training.

# In[21]:


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


# ### Replacing Nulls
# 
# Used in final predicting.

# In[26]:


def replaceNulls(modelFeatures):
    modelFeatures.fillna(value=0, axis=0, inplace=True)    
    return modelFeatures


# ### Calculating distance between partner and customer

# In[27]:


# Calculates distance between coordinates
def calcDist(modelFeatures):       
    modelFeatures['DistanceKM'] = modelFeatures.apply(lambda x: distance.distance( (x['CustomerLatitude'],x['CustomerLongitude']) , (x['PartnerLatitude'],x['PartnerLongitude']) ).km, axis=1)
    return modelFeatures


# In[28]:


mf = calcDist(mf)


# In[29]:


# Verifying function
mf.plot.scatter(x='DistanceKM',y='DeliveryTime')


# In[30]:


# Distance Histogram
mf.DistanceKM.plot.hist(bins=50)


# ### Preparing categorical variables

# In[31]:


# Preparing categorical variables for one hot encoding
def prepareEncoding(modelFeatures):
    modelFeatures['DeliveryType'] = modelFeatures.DeliveryType.apply(str).apply(lambda x: 1 if x == 'Express' else 0)
    modelFeatures['DdpCategory'] = modelFeatures.DdpCategory.apply(str).apply(lambda x: 'CategoryClothing' if x == 'Clothing & Accessories' else ('CategoryFootwear' if x[:8] == 'Footwear' else 'CategoryOthers'))
    modelFeatures['Category1stLevel'] = modelFeatures.Category1stLevel.apply(str).apply(lambda x: 'LevelClothing' if x == 'Clothing' else ('LevelShoes' if x == 'Shoes'  else ('LevelBags' if x == 'Bags'  else 'LevelOthers')))
    return modelFeatures


# In[32]:


# Applying changes
mf = prepareEncoding(mf)


# In[33]:


# Verifying changes
mf.head()


# ### Getting month from data

# In[34]:


# Get month from date due to seasonality
def getMonth(modelFeatures):
    modelFeatures['OrderDate'] = modelFeatures.OrderDate.str.slice(5,7)
    modelFeatures.rename(index=str, columns={'OrderDate':'OrderMonth'}, inplace=True)
    return modelFeatures


# In[35]:


# Applying changes
mf = getMonth(mf)


# In[36]:


# Verifying changes
mf.head()


# ### Encoding categorical variables

# In[37]:


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


# In[38]:


# Applying changes
mf = encodeCategories(mf)


# In[39]:


# Verifying changes
mf.head()


# ### Dropping already used features

# In[40]:


# Dropping features we are not going to use
def dropUsed(modelFeatures):
    modelFeatures.drop(labels='DdpCategory', axis=1, inplace=True)
    modelFeatures.drop(labels='Category1stLevel', axis=1, inplace=True)
    
    return modelFeatures


# In[41]:


# Applying changes
mf = dropUsed(mf)


# In[42]:


# Verifying changes
mf.head()


# ## Model Selection
# 
# In this section we are going to use the features that we have prepared, in order to choose the best model.
# 
# We will train the following models:
# - Linear Regression
# - Decision Tree
# - Random Forest
# - Ada Boost
# - Gradient Boosting

# In[43]:


# Creating the dataframe
modelDf = df.copy()


# In[44]:


# Applying feature engineering transformations
modelDf = dropUnused(modelDf)
modelDf = dropNulls(modelDf)
modelDf = calcDist(modelDf)
modelDf = prepareEncoding(modelDf)
modelDf = getMonth(modelDf)
modelDf = encodeCategories(modelDf)
modelDf = dropUsed(modelDf)


# In[45]:


# Verifying changes
modelDf.head()


# In[46]:


# Splitting X and Y
X = modelDf.drop(['DeliveryTime'], axis=1)
y = modelDf['DeliveryTime']

# Scaling X
col_names = ['CustomerLatitude', 'CustomerLongitude','PartnerLatitude','PartnerLongitude','DistanceKM','OrderMonth']
scaler = StandardScaler().fit(X[col_names])
X[col_names] = scaler.transform(X[col_names])
X.head()


# In[47]:


# Splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[48]:


# Definig hyperparameter space
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
                'loss':['linear','square','exponential']},
    
    'GradientBoosting':{'loss':['ls','lad','huber'],
                        'learning_rate':[0.01,0.1,1],
                        'n_estimators':[50,100,200],
                        'criterion':['mse','friedman_mse','mae']}
}


# In[49]:


# Defining models
models = {
    'LinearRegression':LinearRegression(),
    'DecisionTree':DecisionTreeRegressor(),
    'RandomForest':RandomForestRegressor(),
    'AdaBoost':AdaBoostRegressor(),
    'GradientBoosting':GradientBoostingRegressor()
}


# In[50]:


# Defining GridSearch selection helper 
# Source: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
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


# In[51]:


#Executing GridSearch
helper = EstimatorSelectionHelper(models, params)
helper.fit(X_train, y_train, scoring='neg_mean_squared_error', n_jobs=3)


# In[52]:


#Verifying results
helper.score_summary(sort_by='max_score')


# In[53]:


# Analyzing the best result
res = helper.score_summary(sort_by='max_score')


# In[54]:


# Analyzing the best result: parameters
res.iloc[0]['base_estimator']


# In[55]:


# Analyzing the best result: parameters
res.iloc[0]


# ## Training the best model
# 
# We are now going to train the best model, based on our previous search.

# In[56]:


# Defining model
bestModel = GradientBoostingRegressor(
    criterion = 'mse',
    learning_rate = 0.1,
    loss = 'huber',
    n_estimators = 200
)


# In[57]:


# Training model
bestModel.fit(X_train,y_train)


# In[58]:


# Predicting
y_pred = bestModel.predict(X_test)


# In[59]:


# Verifying results
resultDf = X_test.copy()
resultDf['DeliveryTime'] = y_test
resultDf['PredictedTime'] = y_pred


# In[70]:


# Verifyng df
resultDf.head(15)


# ## Saving the model
# 
# Now we are going to save the model, in order to use it in a real pipeline.

# In[61]:


dump(bestModel, 'eddModel.joblib') 


# ## Predicting Test Set
# 
# In this section we are going to use the model we just exported, in order to predict the values in the test set that was provided.
# 
# Then, the results are going to be exported into a CSV file.

# In[62]:


# Loading previously defined model
prodModel = load('eddModel.joblib')


# In[63]:


# Loading test set
testFile = pd.read_csv('data/test.csv')
testDf = pd.DataFrame(testFile)
outputDf = testDf.copy()


# In[64]:


# Applying feature engineering transformations
col_names = ['CustomerLatitude', 'CustomerLongitude','PartnerLatitude','PartnerLongitude','DistanceKM','OrderMonth']
testDf = dropUnused(testDf)
testDf = replaceNulls(testDf)
testDf = calcDist(testDf)
testDf = prepareEncoding(testDf)
testDf = getMonth(testDf)
testDf = encodeCategories(testDf)
testDf = dropUsed(testDf)
scaler = StandardScaler().fit(testDf[col_names])
testDf[col_names] = scaler.transform(testDf[col_names])


# In[65]:


# Predicting
testPred = prodModel.predict(testDf)


# In[66]:


# Preparing output
outputDf['DeliveryTime'] = testPred


# In[67]:


# Exporting result
outputDf.to_csv(path_or_buf='edd_results.csv', columns=['OrderLineID','DeliveryTime'], index=False)


# ## Requirements
# 
# List of used packages and versions.

# In[68]:


import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to had
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))
        
requirements

