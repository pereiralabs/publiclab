
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')


# # Distribuição de Probabilidade

# ### Introdução
# 
# Neste tutorial você vai:
# 
# - Aprender alguns jargões de probabilidade
# - Aprender diferentes tipos de distribuição de probabilidade
# - Aprender a criar e plotar estas distribuições em Python

# ### Variável aleatória discreta
# 
# Uma variável aleatória é uma variável cujos valores numéricos são representações de fenômenos aleatórios. Existem dois tipos de variáveis aleatórias:
# 
# Uma **variável aleatória discreta** é uma variável que assume valores distintos e limitados. Um exemplo é uma variável X que representa os valores de lançamento de um dado, neste cenário X representa os valores [1, 2, 3, 4, 5, 6].
# 
# A **distribuição de probabilidade** de uma variável discreta é uma **lista de probabilidades** associada a cada possível valor da variável discreta.
# 
# Sendo *pi* a probilidade de *X=xi*, temos que *P(X=xi) = pi*, com *pi* tendo que respeitar duas condições:
# 
# Primeira condição:
# 0 < *pi* < 1 para cada *i*
# 
# Segunda condição:
# p1 + p2 + ... + pn = 1
# 
# Exemplo:

# In[2]:


#Primeira condição: P(X=1) = p1
p1 = 1/6

p1


# In[3]:


#Segunda condição: p1 + p2 + p3 + p4 + p5 + p6 = 1
p1 = 1/6
p2 = 1/6
p3 = 1/6
p4 = 1/6
p5 = 1/6
p6 = 1/6

R = p1 + p2 + p3 + p4 + p5 + p6

R


# ### Variável aleatória contínua
# 
# Uma **variável aleatória contínua** assume uma quantidade infinita de valores numéricos. Por exemplo: uma variável X que assume a altura, em centímetros, das pessoas.
# 
# A **distribuição de probabilidade** de uma variável aleatória contínua é uma **função de distribuição de probablidade**. A probabilidade de se observar qualquer valor específico é de 0, já que a função pode assumir valores infinitos. Portanto, a probabilidade de *X* estar em um conjunto de valores *A* é representada por *P(A)* e pela **área abaixo da curva**.
# 
# A curva que representa *p(x)* é dada por:
# 
# Primeira condição:
# A curva não possui valores negativos, portanto, *p(x) > 0, para todo x*
# 
# Segundação condição:
# A área total abaixo da curva é igual a 1.
# 
# Esta curva também pode receber o nome de **curva de densidade**.
# 
# Veremos exemplos ao longo do notebook.

# ### Bibliotecas Python
# 
# Vamos agora importar algumas bibliotecas necessárias ao nosso trabalho.

# In[4]:


# import matplotlib
import matplotlib.pyplot as plt

# for latex equations
from IPython.display import Math, Latex

# for displaying images
from IPython.core.display import Image


# In[5]:


# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})


# ## Tipos de distribuição probabilística
# 
# Veremos, agora, alguns tipos de distribuição.

# ### 1. Distribuição Uniforme
# 
# Provavelmente a distribuição mais simples. Nela, todas as possibilidades possuem a mesma probabilidade de acontecer, sendo assim, a distribuição é graficamente um retângulo:
# 
# ![alt text](images/uniform_graph.png)
# 
# 
# E é definida matematicamente pela seguinte função:
# ![alt text](images/uniform_formula.png)
# 
# 
# Vamos verificar como criar esta distribuição em Python:

# In[10]:


#Biblioteca para geração de dados uniformes
from scipy.stats import uniform

#Definição dos parâmetros da função de geração de dados uniformes
n = 1000
start = 10
width = 20
random_state=42
data_uniform = uniform.rvs(size=n, loc=start, scale=width, random_state=random_state)


# In[12]:


#Agora vamos plotar os dados
ax = sns.distplot(
    data_uniform,
    bins=10,
    kde=True,
    color='skyblue',
    hist_kws={'linewidth': 15, 'alpha': 1}
)

ax.set(xlabel='Uniform Distribution', ylabel='Frequency')


# ### Distribuição Normal (Gaussiana)
# 
# A distribuição Normal é muito utilizada em Ciência de Dados. 
# 
# Sua curva de densidade possui o formato de um sino, sendo que o meio da curva representa a média da distribuição e sua abertura representa o desvio padrão a partir da média. Ela é uma curva simétrica:
# 
# ![alt text](images/normal_graph.png)
# 
# 
# Um aspecto importate desta distribuição é que podemos estar certos de que:
# - 68,26% das amostras encontram-se a até 1 desvio padrão da média
# - 95,44% das amostras encontram-se a até 2 desvios padrão da média
# - 99,73% das amostras encontram-se a até 3 desvios padrão da média
# 
# Sua definição matemática é dada por:
# ![alt text](images/normal_formula.png)
# 
# 
# Vamos criar uma amostra com distribuição normal e plotá-la:

# In[13]:


#Biblioteca para geração de dados com distribuição normal
from scipy.stats import norm

#Criação de 1000 números aleatórios entre 0 e 1
size = 1000
start = 0
stop = 1
data_normal = norm.rvs(size=size, loc=start, scale=1)


# In[19]:


ax = sns.distplot(
    data_normal,
    kde=True,
    color='red',
    hist_kws={'linewidth': 15, 'alpha': 1}
)

ax.set(xlabel='Normal Distribution', ylabel='Frequency')


# ### Estatísticas da distribuição
# 
# Verificando as estatísticas da distribuição:

# In[ ]:


import numpy as np


# In[57]:


#Média
print("Média: {0:8.4f}".format(data_uniform.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_uniform)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_uniform)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_uniform)))


# In[58]:


#Média
print("Média: {0:8.4f}".format(data_normal.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_normal)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_normal)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_normal)))


# ### Distribuição Gama
# 
# A distribuição Gama é composta por 2 parâmetros (*alpha* e *beta*) e é um tipo de distribuição da qual fazem parte as distribuições: exponencial, chi-quadrado, erlang, etc.
# 
# Sua definição matemática é dada por:
# ![alt text](images/gama_formula.png)
# 
# Uma distribuição gama típica se parece com:
# ![alt text](images/gama_graph.png)
# 
# 
# Agora vamos verificar sua implementação:

# In[27]:


from scipy.stats import gamma
data_gamma = gamma.rvs(a=5, size=10000, random_state=42)


# In[28]:


ax = sns.distplot(data_gamma,
                  kde=True,
                  bins=100,
                  color='green',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Gamma Distribution', ylabel='Frequency')


# ### Estatísticas da distribuição
# 
# Verificando as estatísticas da distribuição:

# In[56]:


#Média
print("Média: {0:8.4f}".format(data_normal.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_normal)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_normal)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_normal)))


# In[55]:


#Média
print("Média: {0:8.4f}".format(data_gamma.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_gamma)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_gamma)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_gamma)))


# ### Distribuição Exponencial
# 
# A distribuição exponencial descreve eventos que são afetados por uma taxa determinada pelo expoente x na fórmula:
# 
# ![alt text](images/exponential_formula.png)
# 
# 
# Por causa do efeito exponencial, uma curva decrescente se parecerá com:
# 
# ![alt text](images/exponential_graph.png)
# 
# 
# Para simularmos esta distribuição em Python, execute o código abaixo:

# In[59]:


from scipy.stats import expon
data_expon = expon.rvs(scale=1, loc=0, size=1000)


# In[62]:


#Agora vamos plotar os dados
ax = sns.distplot(
    data_expon,
    kde=True,
    bins=10,
    color='skyblue',
    hist_kws={'linewidth': 15, 'alpha': 1}
)

ax.set(xlabel='Exponential Distribution', ylabel='Frequency')


# ### Estatísticas da distribuição
# 
# Verificando as estatísticas da distribuição:

# In[63]:


#Média
print("Média: {0:8.4f}".format(data_normal.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_normal)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_normal)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_normal)))


# In[64]:


#Média
print("Média: {0:8.4f}".format(data_expon.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_expon)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_expon)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_expon)))


# ### Distribuição Binomial
# 
# Uma distribuição onde apenas dois outputs são possíveis (ex.: sucesso ou falha, perda ou ganho, vitória ou derrota, etc) e onde a probabilidade de cada opção é a mesma em todas as tentativas é chamada de Distribuição Binomial.
# 
# É importante ressaltar que, embora as probabilidades sejam iguais, os resultados não precisam ser proporcionalmente distribuídos, mesmo levando em conta a independência de cada rodada.
# 
# Os parâmetros de uma Distribuição Binomial são *n* que indica a quantidade de jogadas/rodadas e *p* que indica a probabilidade de sucesso de cada uma. A função de probabilidade é dada por:
# 
# ![alt text](images/binomial_formula.png)
# 
# 
# Agora vamos gerar esta distribuição:

# In[66]:


from scipy.stats import binom
data_binom = binom.rvs(n=10,p=0.8,size=10000)


# In[67]:


ax = sns.distplot(data_binom,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')


# ### Estatísticas da distribuição
# 
# Verificando as estatísticas da distribuição:

# In[68]:


#Média
print("Média: {0:8.4f}".format(data_normal.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_normal)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_normal)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_normal)))


# In[69]:


#Média
print("Média: {0:8.4f}".format(data_binom.mean()))

#Mediana
print("Mediana: {0:8.4f}".format(np.median(data_binom)))

#Desvio padrão
print("Desvio padrão: {0:8.4f}".format(np.std(data_binom)))

#Variância
print("Variância: {0:8.4f}".format(np.var(data_binom)))

