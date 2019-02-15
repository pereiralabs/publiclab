# Wine Classification

O dataset consiste em features que nos ajudarão a prever a qualidade de um vinho. A qualidade é medida por uma nota de 0 a 10, porém não de maneira contínua, ou seja, não existem notas fracionárias, apenas notas inteiras.

Esta característica faz com que o problema possa ser abordado como um problema de classificação, ao invés de um problema de regressão, pois temos 11 classes que são as notas entre 0 e 10 [0,1,2,3,4,5,6,7,8,9,10].

Ao tratar o problema como um problema de classificação, foi utilizada a acurácia normalizada para validação dos modelos. A acurácia normalizada (ou balanceada) é igual à acurácia básica, porém ela dá o mesmo peso para todas as classes, de forma que a acurácia final é uma média da acurácia de todas as classes, o que é mais correto para analisar a eficiência dos modelos.

Foram utilizados algoritmos reconhecidos por sua capacidade de classificação: Decision Tree, AdaBoost com Decision Tree e Random Forest.

O projeto foi desenvolvido em Jupyter Notebook, na linguagem Python, o notebook ``wine_classification.ipynb`` apresenta mais detalhes das decisões tomadas, assim como a análise dos dados do dataset e o resultado da acurácia.
