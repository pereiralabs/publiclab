
# coding: utf-8

# In[1]:


#Imports necessários
import collections
import numpy as np
import pandas as pds

from imblearn.over_sampling import SMOTE
from pandas import concat
from sklearn import preprocessing


# In[2]:


#Análise do dataset
data = pds.read_csv('medical.csv')
print(len(data))
data.head()
#data.describe()
#collections.Counter(data['No-show'])
#data.isnull().any()


# In[3]:


#Transformação dos dados
data['ScheduledMonth'] = data.ScheduledDay.str.slice(5,7)
data.ScheduledMonth = preprocessing.scale(data.ScheduledMonth)
data['ScheduledMonthDay'] = data.ScheduledDay.str.slice(8,10)
data.ScheduledMonthDay = preprocessing.scale(data.ScheduledMonthDay)
data.ScheduledDay = pds.to_datetime(data.ScheduledDay.str.slice(0,10))
data.AppointmentDay = pds.to_datetime(data.AppointmentDay.str.slice(0,10))
data.ScheduledDay = abs((data.ScheduledDay - data.AppointmentDay).dt.days)
data.rename(columns={'ScheduledDay':'WaitingDays'}, inplace=True)

data = data.drop('AppointmentDay',1)
data = data.drop('Neighbourhood',1)
data = data.drop('PatientId',1)
data = data.drop('AppointmentID',1)
data.rename(columns={'No-show':'Noshow'}, inplace=True)

data.Age = pds.to_numeric(data.Age)
data.Age = preprocessing.scale(data.Age)
data.Noshow = data.Noshow.map({'Yes': 1, 'No': 0})
data.Gender = data.Gender.map({'F': 1, 'M': 0})
data = data[data.Age > 0]
data.head()


# In[4]:


#Verificação de dados do dataset
collections.Counter(data.Noshow)

#Imbalancement treatment com under sampling
#dataNoShow = data.loc[data['Noshow']==1]
#collections.Counter(dataNoShow.Noshow)

#dataShow = data.loc[data['Noshow']==0]
#collections.Counter(dataShow.Noshow)

#data = concat([dataShow.sample(n=25000),dataNoShow])
#collections.Counter(data.Noshow)


# In[5]:


#Separação do dataset
from sklearn.model_selection import train_test_split

X = data.drop('Noshow',1)
y = data.Noshow

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


# In[6]:


#Imbalancement treatment com over sampling
balancer = SMOTE()

x_resampled, y_resampled = balancer.fit_sample(X_train, y_train)

print(collections.Counter(y_train))

print(collections.Counter(y_resampled))


# In[7]:


#Criação dos modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  

rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', random_state=0)
lr = LogisticRegression(random_state=0, penalty='l2', C=1, fit_intercept=True, solver='liblinear')
dt = DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='best')  

algs = []
algs.append(rf)
algs.append(lr)
algs.append(dt)

for alg in algs:
    print('Fitting: ', type(alg).__name__)    
    alg.fit(x_resampled, y_resampled)  


# In[8]:


#Acurácia do modelo
from sklearn.metrics import accuracy_score, confusion_matrix

for alg in algs:
    predicted = alg.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print('Acc: ', type(alg).__name__, f' = {accuracy:.3}')
    print(confusion_matrix(y_test, predicted))
    print()
    print()

