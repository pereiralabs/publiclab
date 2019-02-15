
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


#Imports necessarios
import collections
import itertools
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

from pandas import concat
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:


#Load dos dados
originalData = pds.read_csv('medical.csv')
print(originalData.shape)
originalData.head()


# In[6]:


def format_data_readable(df):
    """
    Formata os dados para uma versão mais human readable
    
    Parameters:
    -----------
    df: `pandas.DataFrame`. Contains appointment data.
    
    Returns:
    --------
    formated_df: `pandas.DataFrame`. Contains appointment data in a more readable way.
    """
    
    #Verifica se o objeto passado é um pandas.DatafRame
    assert isinstance(df, pds.DataFrame), "Expected df to be a pandas DataFrame object. Got {}.".format(type(df))
    
    #Formatacao de dataset para gráficos
    formated_df = df.copy()
    formated_df.SMS_received = formated_df.SMS_received.map({0: 'No SMS', 1: 'Sms Received'})
    formated_df['No-show'] = formated_df['No-show'].map({'Yes': 'No Show', 'No': 'Show Up'})
    
    return formated_df


# In[7]:


#Formatacao de dataset para gráficos
analysisData = format_data_readable(originalData)
analysisData.head()


# In[10]:


def categorical_to_numerical(df):
    """
    Transforma features categóricas em numéricas
    
    Parameters:
    -----------
    df: `pandas.DataFrame`. Contains appointment data.
    
    Returns:
    --------
    formated_df: `pandas.DataFrame`. Contains appointment data with transformed categorical features.
    """
    
    #Verifica se o objeto passado é um pandas.DatafRame
    assert isinstance(df, pds.DataFrame), "Expected df to be a pandas DataFrame object. Got {}.".format(type(df))
    
    #Mapeia features categóricas para valores numéricos
    formated_df = df.copy()
    formated_df['No-show'] = formated_df['No-show'].map({'Yes': '1', 'No': '0'})
    formated_df['No-show'] = pds.to_numeric(formated_df['No-show'])
    
    return formated_df


# In[11]:


#Correlação de atributos
corrData = categorical_to_numerical(originalData)
df_corr = corrData.corr()
df_corr


# In[26]:


#Relacao de SMS vs Noshow
analysisData.groupby(['SMS_received','No-show']).size().plot(kind='bar')


# In[5]:


#Relacao de Scholarship vs Noshow
analysisData.groupby(['Scholarship','No-show']).size().plot(kind='bar')


# In[6]:


#Relacao de Hipertension vs Noshow
analysisData.groupby(['Hipertension','No-show']).size().plot(kind='bar')


# In[7]:


#Relacao de Diabetes vs Noshow
analysisData.groupby(['Diabetes','No-show']).size().plot(kind='bar')


# In[13]:


def feature_engineering(df):
    """Realiza transformações nos dados para criação das features do modelo
    
    Parameters:
    -----------
    df: `pandas.DataFrame`. Contém os dados de medical appointments.
    
    Returns:
    --------
    formatted_df: `pandas.DataFrame`. Contém as features para o modelo.
    """
    
    #Drop dos IDS
    formatted_df = df.copy()
    formatted_df = formatted_df.drop('PatientId',1)
    formatted_df = formatted_df.drop('AppointmentID',1)

    #Renomeando a coluna No-show
    formatted_df.rename(columns={'No-show':'Noshow'}, inplace=True)

    #Transformacao dos dados
    formatted_df.Noshow = formatted_df.Noshow.map({'Yes': 1, 'No': 0})
    formatted_df.Gender = formatted_df.Gender.map({'F': 1, 'M': 0})
    formatted_df.Age = pds.to_numeric(formatted_df.Age)
    formatted_df.Neighbourhood = formatted_df.Neighbourhood.astype("category").cat.codes

    #Construcao de novas colunas
    formatted_df['SchMonth'] = formatted_df.ScheduledDay.str.slice(5,7)
    formatted_df['SchDay'] = formatted_df.ScheduledDay.str.slice(8,10)

    formatted_df.ScheduledDay = pds.to_datetime(formatted_df.ScheduledDay.str.slice(0,10))
    formatted_df.AppointmentDay = pds.to_datetime(formatted_df.AppointmentDay.str.slice(0,10))
    formatted_df['WaitingDays'] = abs((formatted_df.ScheduledDay - formatted_df.AppointmentDay).dt.days)

    #Drop de colunas
    formatted_df = formatted_df.drop('ScheduledDay',1)
    formatted_df = formatted_df.drop('AppointmentDay',1)
    
    return formatted_df


# In[14]:


def scale_features(df):
    """Realiza a normalização das features do modelo
    
    Parameters:
    -----------
    df: `pandas.DataFrame`. Dataframe com as features do modelo.
    
    Returns:
    --------
    scaled_df: `pandas.DataFrame`. Dataframe com as features normalizadas.
    """
    
    scaled_df = df.copy()
    scaled_df.SchMonth = preprocessing.scale(list(scaled_df.SchMonth.astype(float)))
    scaled_df.SchDay = preprocessing.scale(list(scaled_df.SchDay.astype(float)))
    scaled_df.Age = preprocessing.scale(list(scaled_df.Age.astype(float)))
    scaled_df.Neighbourhood = preprocessing.scale(list(scaled_df.Neighbourhood.astype(float)))
    
    return scaled_df


# In[16]:


data = originalData.copy()

#Cria as features para o modelo
data = feature_engineering(data)

#Normalizacao das features
data = scale_features(data)

#Remocao de dados invalidos
data = data[data.Age > 0]

data.head()


# In[17]:


#Reindexação dos dados
data = data.reindex(columns=sorted(data.columns))
data = data.reindex(columns=(['Noshow'] + list([a for a in data.columns if a != 'Noshow'])))
data.head()


# In[18]:


corr = data.corr()
fig, ax = plt.subplots(figsize=(20, 20))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)


# In[19]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[23]:


def train_val_test_split(X, y, test_size=0.2, random_state=0, over_sampling=False):
    """Gets input data and returns it splitted in three parts (training, validation and testing). Optionally, it can oversample the data.
    
    Parameters:
    -----------
    X: `pandas.DataFrame` or `pandas.Series`. Contains model features.
    y: `pandas.Series`. Contains the model target.
    test_size: `float`. test_size will be passed to sklearn.model_selection.train_test_split function.
    random_state: `integer`. random_state will be passed to sklearn.model_selection.train_test_split function.
    over_sampling: `boolean`. Indicates if data should be oversampled before returning the result.
    
    Returns:
    --------
    X_train = `pandas.DataFrame` or `pandas.Series`. Contains model features. Should be used on the model training phase.
    X_val = `pandas.DataFrame` or `pandas.Series`. Contains model features. Should be used on the model training phase.
    X_test = `pandas.DataFrame` or `pandas.Series`. Contains model features. Should be used for final model assessment.
    y_train = `pandas.Series`. Contains the model target. Should be used on the model training phase.
    y_val = `pandas.Series`. Contains the model target. Should be used on the model training phase.
    y_test = `pandas.Series`. Contains the model target. Should be used for final model assessment.
    """
    
    #Required function
    from sklearn.model_selection import train_test_split
    
    #Splits test from the rest
    X_model, X_test, y_model, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  
    
    #Splits train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size=test_size, random_state=random_state)  
    
    #If over_sampling flag is set, than data is oversampled
    if (over_sampling==True):
        #Required function
        from imblearn.over_sampling import SMOTE
        
        #Creates oversampled set
        balancer = SMOTE(kind='regular')
        X_resampled_train, y_resampled_train = balancer.fit_sample(X_train, y_train)
        X_resampled_val, y_resampled_val = balancer.fit_sample(X_val, y_val)
        X_resampled_test, y_resampled_test = balancer.fit_sample(X_test, y_test)
        
        #Overrides original variables to make returning easier
        X_train = X_resampled_train        
        y_train = y_resampled_train
        X_val = X_resampled_val
        y_val = y_resampled_val
        X_test = X_resampled_test
        y_test = y_resampled_test
        
    #Returns dataset
    return X_train,X_val,X_test,y_train,y_val,y_test
        


# In[38]:


#Separação do dataset
X = data.drop('Noshow',1)
y = data.Noshow
X_train,X_val,X_test,y_train,y_val,y_test = train_val_test_split(X,y,over_sampling=True)


# In[39]:


#Validating the output
print('Normal Data: ', collections.Counter(y))
print('Resampled: ', collections.Counter(y_train))
print('Resampled: ', collections.Counter(y_val))
print('Resampled: ', collections.Counter(y_test))


# In[40]:


#Criacao dos modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  

algsSize = 3

#Algoritimos que serao treinados
algs = []
algs.append(RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto', random_state=0))
algs.append(LogisticRegression(random_state=0, penalty='l2', C=1, fit_intercept=True, solver='liblinear'))
algs.append(DecisionTreeClassifier(random_state=0, criterion='entropy', splitter='best')  )

for x in range(0, algsSize):
    print('Fitting: ', type(algs[x]).__name__)
    algs[x].fit(X_train, y_train)     


# In[42]:


#Definicao de dataframe para exibicao de resultados
results = pds.DataFrame(columns=['Name', 'Type', 'Resampled', 'ACC'])


# In[43]:


#Função para display de resultados
def appendResult(alg, dataType, resampled, X, y):
    algName = type(alg).__name__
    predicted = alg.predict(X)
    accuracy = accuracy_score(y, predicted)
    results.loc[len(results)]=[algName, dataType, resampled, accuracy]
    print('Confusion Matrix - ', algName, ' RESAMPLED = ', resampled)
    plot_confusion_matrix(cm=confusion_matrix(y, predicted), target_names=['Show', 'NoShow'])


# In[44]:


#Acuracia do modelo em VALIDACAO
for x in range(0, algsSize):
    appendResult(algs[x], 'Validation', 'Yes', X_val, y_val)    


# In[45]:


#Acuracia do modelo em TESTE
for x in range(0, algsSize):
    appendResult(algs[x], 'Test', 'Yes', X_test, y_test)    


# In[46]:


#Resultados
results

