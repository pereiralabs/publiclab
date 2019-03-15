
# coding: utf-8

# # INF-616 - Tarefa 1
# 
# Professor: Alexandre Ferreira -- melloferreira@ic.unicamp.br  
# Monitor: Lucas David -- ra188972@students.ic.unicamp.br
# 
# Instituto de Computação - Unicamp  
# 2018
# 
# Alunos:
#     Felipe Pereira
#     Miguel Di Ciurcio Filho

# ## Classificação binária (decisão)

# In[11]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

from __future__ import print_function

get_ipython().magic('matplotlib inline')


# In[2]:


dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                    test_size=0.25,
                                                    random_state=42)

print(dataset.DESCR)


# In[3]:


print('Gráfico exibindo as duas primeiras características do sub-conjunto de treino:')
_ = plt.scatter(x_train[:, 0], x_train[:, 1],
                c=y_train,
                alpha=0.8)
_ = plt.xlabel(dataset.feature_names[0])
_ = plt.ylabel(dataset.feature_names[1])


# In[4]:


estimators = [SVC(random_state=13), LogisticRegression(random_state=24)]

for e in estimators:
    print('Treinando estimator', type(e).__name__)
    e.fit(x_train, y_train)

print('Todos os estimatores foram treinados!')


# ### Qual dos dois estimadores apresenta menor taxa de erro sobre o conjunto WDBC teste?

# In[5]:


#Criação de função para cálculo da Matriz de Confusão
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Matriz de confusão normalizada')
    else:
        print('Matriz de confusão não normalizada')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')


# In[6]:


#Libs para a matriz de confusão - CONJUNTO DE TREINO
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

for e in estimators:
    print('Teste do estimator', type(e).__name__)
    p_train = e.predict(x_train)
    
    print('exemplo de rótulos de treino verdadeiros:', y_train[:10], '...')
    print('exemplo de rótulos de treino preditos:', p_train[:10], '...')
    
    cnf_matrix = confusion_matrix(y_train, p_train)
    np.set_printoptions(precision=2)
    
    plt.figure()
    #class_names = ['not_cancer' , 'cancer']
    class_names = dataset.target_names[:2]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matriz de confusão não normalizada')
    


# - Algum dos estimatores super-especificou sobre o conjunto de treinamento?
#     FP: Sim, o SVC
# - Quantas vezes cada um dos estimadores errou, no conjunto de teste?
#     FP: SVC possui 0 erros. Linear Regression possui 18 erros.

# ### Os estimadores conseguem distinguir ambas as classes de forma satisfatória?

# Utilize uma ou mais funções vistas em aula para descobrir se os classificadores efetivamente conseguem distinguir amostras benignas de malignas.

# In[7]:


#Bibliotecas para análise dos dados
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import interp

for e in estimators:
    print('Estimador', type(e).__name__)
    p_test = e.predict(x_test)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, p_test, target_names=class_names))


# As pontuações mudam quando consideramos a frequência das classes? FP: Sim, principalmente para o SVC, que foi melhor no positivo do que no negativo. O LinearRegression foi mais parecido, com apenas pequenas diferenças de uma classe para outra. Os dados de Treino e Teste estavam igualmente balanceados, com cerca de 1 terço dos dados positivos. Esta diferença pode ter feito o SVC aprender melhor a classe engativa, que tinha 70% dos dados.

# In[10]:


#Libs para a matriz de confusão - CONJUNTO DE TESTE
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

for e in estimators:
    print('Teste do estimator', type(e).__name__)
    p_test = e.predict(x_test)
    
    print('exemplo de rótulos de treino verdadeiros:', y_test[:10], '...')
    print('exemplo de rótulos de treino preditos:', p_test[:10], '...')
    
    cnf_matrix = confusion_matrix(y_test, p_test)
    np.set_printoptions(precision=2)
    
    plt.figure()
    #class_names = ['not_cancer' , 'cancer']
    class_names = dataset.target_names[:2]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Matriz de confusão não normalizada')
    


# ### Apresente um relatório das principais métricas para ambos estimadores

# In[9]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

for e in estimators:
    print('Estimador', type(e).__name__)
    
    p_test = e.predict(x_test)
    
    #Mean Absolute Error 
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test , p_test)))

    #Mean Squared Error    
    print('MSE: {:.2f}'.format(mean_squared_error(y_test , p_test)))
    
    #Acuracia
    from sklearn.metrics import accuracy_score
    print('Acurácia normalizada = {:.2f}'.format(accuracy_score(y_test, p_test)))
    print('Acurácia = {:.0f}'.format(accuracy_score(y_test, p_test, normalize=False)))
    
    #AUC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, p_test)
    print('AUC = {:.2f}'.format(auc(fpr, tpr)))
    
    #F1
    from sklearn.metrics import f1_score
    print('F1 = {:.2f}'.format(f1_score(y_test, p_test, average='binary')))
    
    #Espaço em branco
    print(' ')
    


# Qual estimador possui melhor *f-1 score*?  FP: O LogisticRegression com 0.97, pois indica que possui melhor harmonia entre o precision e o recall, tornando o resultado do algoritmo melhor.
# 
