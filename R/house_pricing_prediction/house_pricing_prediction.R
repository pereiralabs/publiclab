#Limpeza de cache
rm(list=ls())

#PASSOS A SEREM REALIZADOS
  #1. Verificar quantos elementos temos (OK)
  #2. Verificar se existe elementos faltantes em cada feature dos dados de treinamento e dados de validaçao. Remover a s?rie completa 
  #3. Transformar os dados da feature discreta (ocean_proximity) para valores num?ricos: <1H = 1, INLAND = 2, ISLAND = 3, NEAR = 4
  #4. Realizar a normalizaçao dos dados para cada feature (OK)
  #Para cada Modelo:
  #5. Calcular LM para dados de treinamento
  #6. Calcular PREDICT para dados de validaçao
  #7. Calcular MAE para dados de validaçao
  #8. Fazer o print da funçao de custo e comparar com os modelos

#MODELOS IMPLEMENTADOS:
  ## Modelo 1 - Todas as features dispon?veis SEM normalizaçao
  ## Modelo 2 - Todas as features dispon?veis COM normalizaçao
  ## Modelo 3 - Reduzir a quantidade de features descartando algumas 
  ## Modelo 4 - Pegar as features do modelo 2 e realizar algum tipo de combinaçao entre elas
  

########################################
# Auxiliary code for Linear Regression #
########################################

# Cost function J(x,y,theta)
cost <- function(X, y, theta) {
  sum( (X %*% theta - y)^2 ) / (2*length(y))
}


########### Gradient Descent
# -- Inputs
# X: matrix MxN (M examples and N features)
# y: target value for each sample - matrix Mx1 (M examples)
# learningRate
# nIter: number of iterations to perform GD
# plotCost: plot the cost value in each iteration
#
# -- Output
# theta: the weights of intercept + each feature (matrix 1 x N+1)
# cost_history: the values of the cost function in each iteration
GD <- function(X, y, learningRate=0.1, nIter=1000, plotCost=FALSE){
  X = cbind(1, X) # add a collum for the intercept in X
  
  nFeatures = dim(X)[2]
  
  theta = matrix(runif(nFeatures), nrow=nFeatures)
  
  #store the cost function values to plot later
  cost_history = matrix(0, ncol=nIter, nrow=1)
  
  
  #### ---> Perform GD
  for (i in 1:nIter) {
    error = (X %*% theta - y)
    delta_J = t(X) %*% error / length(y)
    theta = theta - learningRate * delta_J  #Update all thetas at once
    
    #Storing the cost function value for current iteration
    cost_history[i] = cost(X, y, theta)
  }
  
  return(list(theta=theta, cost_history=cost_history))
} 

  ########## Prediction
  # -- Inputs
  # X: matrix MxN (M examples and N features)
  # theta: weights of the intercept + features
  # -- Outputs
  # prediction: the estimated target value for X
  predict_0615 <- function(X, theta){
    X = cbind(1,X)
    prediction = X %*% theta
    return(prediction)
  }

  
  ########### Normal Equations
  # -- Inputs
  # X: matrix MxN (M examples and N features)
  # y: target value for each sample - matrix Mx1 (M examples)
  # -- Outputs
  # theta: the weights of intercept + each feature (matrix 1 x N+1)
  NE <- function(X, y){
    X = cbind(1,X)
    invXX = solve(t(X) %*% X) 
    theta = invXX %*% (t(X) %*% y)
    return(theta)
  }
  
###############



#0 - Importar os dados
###############

  housePricing_trainSet <- read.csv("/home/felipe/Downloads/housePricing_trainSet.csv", header = TRUE, sep = ",", stringsAsFactors = F) 
  housePricing_valSet <- read.csv("/home/felipe/Downloads/housePricing_valSet.csv", header = TRUE, sep = ",", stringsAsFactors = F)

###############
  
#1
###############

  #Dimensionar os dados de treinamento
  train_elements <- dim(housePricing_trainSet)[1]
  train_features <- dim(housePricing_trainSet)[2]
  print(paste("Quantidade de registros de treino: ",train_elements))
  print(paste("Quantidade de atributos de treino: ",train_features))
  
  #Dimensionar os dados de validaçao
  val_elements <- dim(housePricing_valSet)[1]
  val_features <- dim(housePricing_valSet)[2]
  print(paste("Quantidade de registros de valida?ao: ",val_elements))
  print(paste("Quantidade de atributos de valida?ao: ",val_features))
  
#2
###############

  #Apenas 1% dos registros de treino e validacao sao NA, portanto iremos remove-los
  housePricing_trainSet <-  housePricing_trainSet[!is.na(housePricing_trainSet[,5]),] 
  housePricing_valSet <-  housePricing_valSet[!is.na(housePricing_valSet[,5]),] 
  
  train_cleanSet <-dim(housePricing_trainSet)[1] 
  val_cleanSet <-dim(housePricing_valSet)[1] 
  
  print(paste("Quantidade de registros de treino ap?s limpeza: ",train_cleanSet))
  print(paste("Quantidade de registros de validacao ap?s limpeza: ",val_cleanSet))

###############

#3
###############
  #Transformar os dados da feature discreta (ocean_proximity) para valores num?ricos: 
  #<1H = 1, INLAND = 2, ISLAND = 3, NEAR = 4, NEAR OCEAN = 5
  
  #Funçao que classifica numericamente os atributos discretos de OCEAN PROXIMITY
  ocean_prox_classifier <- function(dataframe){
    i <- 1
    rows <- dim(dataframe)[1]
    ocean_prox_class <- c(1:rows)
    
    for (element in dataframe[,10]) {
      
      if (element == "<1H OCEAN"){
        ocean_prox_class[i] <- 1
      }
      else {
        if (element == "INLAND"){
          ocean_prox_class[i] <- 2  
        }
        else{
          if (element == "ISLAND"){
            ocean_prox_class[i] <- 3
          }
          else{
            if (element == "NEAR BAY"){
              ocean_prox_class[i] <- 4
            }
            else{
              if (element == "NEAR OCEAN"){
                ocean_prox_class[i] <- 5
              }
            }
          }
        }
      }
      
      #Aux counter   
      i <- i + 1
    }
    
    return(ocean_prox_class)
  }
  
  #Obtendo a transformaçao do atributo ocean_prox para os dados de treino
  train_ocean_prox_class <- ocean_prox_classifier(housePricing_trainSet)
  housePricing_trainSet$ocean_prox_class <- train_ocean_prox_class
  
  #Obtendo a transformaçao do atributo ocean_prox para os dados de validaçao
  val_ocean_prox_class <- ocean_prox_classifier(housePricing_valSet)
  housePricing_valSet$ocean_prox_class <- val_ocean_prox_class
  
###############

#4
###############

  # Normalizar dados de treinamento
  #Calculo da media e desvio padrao de cada feature
  mean_value <-mean(housePricing_trainSet$median_house_value)
  dp_value <- sd(housePricing_trainSet$median_house_value)
  
  mean_longitude <-mean(housePricing_trainSet$longitude)
  dp_longitude <- sd(housePricing_trainSet$longitude)
  
  mean_latitude <-mean(housePricing_trainSet$latitude)
  dp_latitude <- sd(housePricing_trainSet$latitude)
  
  mean_house_age <-mean(housePricing_trainSet$housing_median_age)
  dp_house_age <- sd(housePricing_trainSet$housing_median_age)
  
  mean_total_rooms <-mean(housePricing_trainSet$total_rooms)
  dp_total_rooms <- sd(housePricing_trainSet$total_rooms)
  
  mean_total_bedrooms <-mean(housePricing_trainSet$total_bedrooms)
  dp_total_bedrooms <- sd(housePricing_trainSet$total_bedrooms)
  
  mean_population <-mean(housePricing_trainSet$population)
  dp_population <- sd(housePricing_trainSet$population)
  
  mean_households <-mean(housePricing_trainSet$households)
  dp_households <- sd(housePricing_trainSet$households)
  
  mean_income <-mean(housePricing_trainSet$median_income)
  dp_income <- sd(housePricing_trainSet$median_income)
  
  mean_ocean_prox_class <-mean(housePricing_trainSet$ocean_prox_class)
  dp_ocean_prox_class <- sd(housePricing_trainSet$ocean_prox_class)
  
  #normalizaçao de cada feature
  value_normal<-c((housePricing_trainSet$median_house_value-mean_value)/dp_value)
  longitude_normal<-c((housePricing_trainSet$longitude-mean_longitude)/dp_longitude)
  latitude_normal<-c((housePricing_trainSet$latitude-mean_latitude)/dp_latitude)
  house_age_normal<-c((housePricing_trainSet$housing_median_age-mean_house_age)/dp_house_age)
  total_rooms_normal<-c((housePricing_trainSet$total_rooms-mean_total_rooms)/dp_total_rooms)
  total_bedrooms_normal<-c((housePricing_trainSet$total_bedrooms-mean_total_bedrooms)/dp_total_bedrooms)
  population_normal<-c((housePricing_trainSet$population-mean_population)/dp_population)
  households_normal<-c((housePricing_trainSet$households-mean_households)/dp_households)
  income_normal<-c((housePricing_trainSet$median_income-mean_income)/dp_income)
  ocean_prox_class_normal<-c((housePricing_trainSet$ocean_prox_class-mean_ocean_prox_class)/dp_ocean_prox_class)
  
  
  housePricing_trainNormal<-data.frame(value_normal, longitude_normal, latitude_normal, house_age_normal, total_rooms_normal, total_bedrooms_normal, population_normal, households_normal, income_normal, ocean_prox_class_normal)
  
  ###############################################
  
  # Normalizar dados de validaçao
  #Calculo da media e desvio padrao de cada feature
  val_mean_value <-mean(housePricing_valSet$median_house_value)
  val_dp_value <- sd(housePricing_valSet$median_house_value)
  
  val_mean_longitude <-mean(housePricing_valSet$longitude)
  val_dp_longitude <- sd(housePricing_valSet$longitude)
  
  val_mean_latitude <-mean(housePricing_valSet$latitude)
  val_dp_latitude <- sd(housePricing_valSet$latitude)
  
  val_mean_house_age <-mean(housePricing_valSet$housing_median_age)
  val_dp_house_age <- sd(housePricing_valSet$housing_median_age)
  
  val_mean_total_rooms <-mean(housePricing_valSet$total_rooms)
  val_dp_total_rooms <- sd(housePricing_valSet$total_rooms)
  
  val_mean_total_bedrooms <-mean(housePricing_valSet$total_bedrooms)
  val_dp_total_bedrooms <- sd(housePricing_valSet$total_bedrooms)
  
  val_mean_population <-mean(housePricing_valSet$population)
  val_dp_population <- sd(housePricing_valSet$population)
  
  val_mean_households <-mean(housePricing_valSet$households)
  val_dp_households <- sd(housePricing_valSet$households)
  
  val_mean_income <-mean(housePricing_valSet$median_income)
  val_dp_income <- sd(housePricing_valSet$median_income)
  
  val_mean_ocean_prox_class <-mean(housePricing_valSet$ocean_prox_class)
  val_dp_ocean_prox_class <- sd(housePricing_valSet$ocean_prox_class)
  
  #normalizaçao de cada feature
  value_normal<-c((housePricing_valSet$median_house_value-val_mean_value)/val_dp_value)
  longitude_normal<-c((housePricing_valSet$longitude-val_mean_longitude)/val_dp_longitude)
  latitude_normal<-c((housePricing_valSet$latitude-val_mean_latitude)/val_dp_latitude)
  house_age_normal<-c((housePricing_valSet$housing_median_age-mean_house_age)/val_dp_house_age)
  total_rooms_normal<-c((housePricing_valSet$total_rooms-val_mean_total_rooms)/val_dp_total_rooms)
  total_bedrooms_normal<-c((housePricing_valSet$total_bedrooms-val_mean_total_bedrooms)/val_dp_total_bedrooms)
  population_normal<-c((housePricing_valSet$population-val_mean_population)/val_dp_population)
  households_normal<-c((housePricing_valSet$households-val_mean_households)/val_dp_households)
  income_normal<-c((housePricing_valSet$median_income-val_mean_income)/val_dp_income)
  ocean_prox_class_normal<-c((housePricing_valSet$ocean_prox_class-val_mean_ocean_prox_class)/val_dp_ocean_prox_class)
  
  
  val_housePricing_normal<-data.frame(value_normal, longitude_normal, latitude_normal, house_age_normal, total_rooms_normal, total_bedrooms_normal, population_normal, households_normal, income_normal, ocean_prox_class_normal)
  

###############################################



###MODELO 1###
print ("  ")
print ("  ")
print("###############################################")
print("# Modelo 01 - Todas as features              ##")
print("###############################################")
  #5
  #Calculo da regressao linear (lm)
  r_lm_01 <- lm(formula=median_house_value~longitude+latitude+housing_median_age+total_rooms+total_bedrooms+population+households+median_income+ocean_proximity,data=housePricing_trainSet)
  
  #6
  #Predict
  p_01 <- predict (r_lm_01,housePricing_valSet)
  
  #7
  #Calculo de erro medio absoluto (MAE)
  housePricing_valSet$median_house_value_prediction_01 <- p_01
  housePricing_valSet$error_p01 <- abs(housePricing_valSet$median_house_value - housePricing_valSet$median_house_value_prediction_01)
  mean_error_p01 <- mean(housePricing_valSet$error_p01)
  #print(paste("MAE do Modelo 01: ",mean_error_p01))
  
  
  #9A
  #Calculo do GD
  gd_01 <- GD(as.matrix(housePricing_trainSet[,c(1:8,11)]), as.matrix(housePricing_trainSet[,9]), learningRate=0.001, nIter=1000, plotCost=FALSE)
  
  #9B
  #Predicao com GD
  gd_p_01 <- predict_0615(as.matrix(housePricing_valSet[,c(1:8,11)]), gd_01$theta) 
  
  #9C 
  #Erro médio com GD
  housePricing_valSet$median_house_value_prediction_01_gd <- gd_p_01
  housePricing_valSet$error_p01_gd <- abs(housePricing_valSet$median_house_value - housePricing_valSet$median_house_value_prediction_01_gd)
  mean_error_p01_gd <- mean(housePricing_valSet$error_p01_gd)
  
  #9D
  #Plot da função de custo calculada
  #plot(c(1:1000),gd_01$cost_history)
  
  #10A
  #Calculo do NE
  ne_01 <- NE(as.matrix(housePricing_trainSet[,c(1:8,11)]), as.matrix(housePricing_trainSet[,9]))
  
  #10B
  #Predicao com NE
  ne_p_01 <- predict_0615(as.matrix(housePricing_valSet[,c(1:8,11)]), ne_01) 
  
  #10C 
  #Erro médio com NE
  housePricing_valSet$median_house_value_prediction_01_ne <- ne_p_01
  housePricing_valSet$error_p01_ne <- abs(housePricing_valSet$median_house_value - housePricing_valSet$median_house_value_prediction_01_ne)
  mean_error_p01_ne <- mean(housePricing_valSet$error_p01_ne)

###############  

###MODELO 2###
print ("  ")
print ("  ")
print("###############################################")
print("# Modelo 02 - Todas as features normalizadas ##")
print("###############################################")
  #5
  #Calculo da regressao linear (lm)
  r_lm_02 <- lm(formula=value_normal~longitude_normal+latitude_normal+house_age_normal+total_rooms_normal+total_bedrooms_normal+population_normal+households_normal+income_normal+ocean_prox_class_normal,data=housePricing_trainNormal)
  
  #6
  #Predict
  p_02 <- predict (r_lm_02,val_housePricing_normal)
  
  
  #7
  #Calculo de erro medio absoluto (MAE)
  val_housePricing_normal$median_house_value_prediction_02 <- p_02
  val_housePricing_normal$error_p02 <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_02)
  mean_error_p02 <- mean(val_housePricing_normal$error_p02)
  #print(paste("MAE do Modelo 02: ",mean_error_p02))
  
  #9A
  #Calculo do GD
  gd_02 <- GD(as.matrix(housePricing_trainNormal[,c(2:10)]), as.matrix(housePricing_trainNormal[,1]), learningRate=0.1, nIter=20000, plotCost=FALSE)
  
  #9B
  #Predicao com GD
  gd_p_02 <- predict_0615(as.matrix(val_housePricing_normal[,c(2:10)]), gd_02$theta) 
  
  #9C 
  #Erro médio com GD
  val_housePricing_normal$median_house_value_prediction_02_gd <- gd_p_02
  val_housePricing_normal$error_p02_gd <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_02_gd)
  mean_error_p02_gd <- mean(val_housePricing_normal$error_p02_gd)
  
  #9D
  #Plot da função de custo calculada
  #plot(c(1:1000),gd_02$cost_history)
  
  
  #10A
  #Calculo do NE
  ne_02 <- NE(as.matrix(housePricing_trainNormal[,c(2:10)]), as.matrix(housePricing_trainNormal[,1]))
  
  #10B
  #Predicao com NE
  ne_p_02 <- predict_0615(as.matrix(val_housePricing_normal[,c(2:10)]), ne_02) 
  
  #10C 
  #Erro médio com NE
  val_housePricing_normal$median_house_value_prediction_02_ne <- ne_p_02
  val_housePricing_normal$error_p02_ne <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_02_ne)
  mean_error_p02_ne <- mean(val_housePricing_normal$error_p02_ne)
  
###############
  
###MODELO 3###
  print ("  ")
  print ("  ")
  print("###############################################")
  print("# Modelo 03 - Reduçao de atributos           ##")
  print("###############################################")
  #5
  #Calculo da regressao linear (lm)
  r_lm_03 <- lm(formula=value_normal~house_age_normal+total_bedrooms_normal+income_normal+ocean_prox_class_normal,data=housePricing_trainNormal)

  
  #6
  #Predict
  p_03 <- predict (r_lm_03,val_housePricing_normal)
  
  
  #7
  #Calculo de erro medio absoluto (MAE)
  val_housePricing_normal$median_house_value_prediction_03 <- p_03
  val_housePricing_normal$error_p03 <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_03)
  mean_error_p03 <- mean(val_housePricing_normal$error_p03)
  #print(paste("MAE do Modelo 03: ",mean_error_p03))
  
  
  #9A
  #Calculo do GD
  gd_03 <- GD(as.matrix(housePricing_trainNormal[,c(4,5,6,10)]), as.matrix(housePricing_trainNormal[,1]), learningRate=0.1, nIter=1000, plotCost=FALSE)
  
  #9B
  #Predicao com GD
  gd_p_03 <- predict_0615(as.matrix(val_housePricing_normal[,c(4,5,6,10)]), gd_03$theta) 
  
  #9C 
  #Erro médio com GD
  val_housePricing_normal$median_house_value_prediction_03_gd <- gd_p_03
  val_housePricing_normal$error_p03_gd <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_03_gd)
  mean_error_p03_gd <- mean(val_housePricing_normal$error_p03_gd)
  
  #9D
  #Plot da função de custo calculada
  #plot(c(1:1000),gd_03$cost_history)
  
  #10A
  #Calculo do NE
  ne_03 <- NE(as.matrix(housePricing_trainNormal[,c(4,5,6,10)]), as.matrix(housePricing_trainNormal[,1]))
  
  #10B
  #Predicao com NE
  ne_p_03 <- predict_0615(as.matrix(val_housePricing_normal[,c(4,5,6,10)]), ne_03) 
  
  #10C 
  #Erro médio com NE
  val_housePricing_normal$median_house_value_prediction_03_ne <- ne_p_03
  val_housePricing_normal$error_p03_ne <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_03_ne)
  mean_error_p03_ne <- mean(val_housePricing_normal$error_p03_ne)
  
###############  
  
###MODELO 4###
  print ("  ")
  print ("  ")
  print("###############################################")
  print("# Modelo 04 - Features combinadas            ##")
  print("###############################################")
  #5
  #Calculo da regressao linear (lm)
  r_lm_04 <- lm(formula=value_normal~longitude_normal*latitude_normal+house_age_normal+total_rooms_normal*total_bedrooms_normal+population_normal*households_normal*income_normal+ocean_prox_class_normal,data=housePricing_trainNormal)
  
  #6
  #Predict
  p_04 <- predict (r_lm_04,val_housePricing_normal)
  
  
  #7
  #Calculo de erro medio absoluto (MAE)
  val_housePricing_normal$median_house_value_prediction_04 <- p_04
  val_housePricing_normal$error_p04 <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_04)
  mean_error_p04 <- mean(val_housePricing_normal$error_p04)
  #print(paste("MAE do Modelo 04: ",mean_error_p04))
  
  
  #9A
  #Calculo do GD
  housePricing_trainNormal$fc_01 <- housePricing_trainNormal$longitude_normal * housePricing_trainNormal$latitude_normal
  housePricing_trainNormal$fc_02 <- housePricing_trainNormal$total_rooms_normal * housePricing_trainNormal$total_bedrooms_normal
  housePricing_trainNormal$fc_03 <- housePricing_trainNormal$population_normal * housePricing_trainNormal$households_normal * housePricing_trainNormal$income_normal
  gd_04 <- GD(as.matrix(housePricing_trainNormal[,c(11,4,12,13,10)]), as.matrix(housePricing_trainNormal[,1]), learningRate=0.001, nIter=1000, plotCost=FALSE)
  
  #9B
  #Predicao com GD
  gd_p_04 <- predict_0615(as.matrix(val_housePricing_normal[,c(11,4,12,13,10)]), gd_04$theta) 
  
  #9C 
  #Erro médio com GD
  val_housePricing_normal$median_house_value_prediction_04_gd <- gd_p_04
  val_housePricing_normal$error_p04_gd <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_04_gd)
  mean_error_p04_gd <- mean(val_housePricing_normal$error_p04_gd)
  
  #9D
  #Plot da função de custo calculada
  #plot(c(1:1000),gd_04$cost_history)
  
  #10A
  #Calculo do NE
  ne_04 <- NE(as.matrix(housePricing_trainNormal[,c(11,4,12,13,10)]), as.matrix(housePricing_trainNormal[,1]))
  
  #10B
  #Predicao com NE
  ne_p_04 <- predict_0615(as.matrix(val_housePricing_normal[,c(11,4,12,13,10)]),ne_04) 
  
  #10C 
  #Erro médio com NE
  val_housePricing_normal$median_house_value_prediction_04_ne <- ne_p_04
  val_housePricing_normal$error_p04_ne <- abs(val_housePricing_normal$value_normal - val_housePricing_normal$median_house_value_prediction_04_ne)
  mean_error_p04_ne <- mean(val_housePricing_normal$error_p04_ne)
  
########################################################################
######################    Resumos       ################################
########################################################################  


#Resumo das Fun?oes de Custos
  #print(" ")
  #print(" ")
  #print("Resumo das Fun?oes de Custos:")
  #print(paste("Modelo 01: "))
  #print(coe_01)
  #print(paste("Modelo 02: "))
  #print(coe_02)
  #print(paste("Modelo 03: "))
  #print(coe_03)
  #print(paste("Modelo 04: "))
  #print(coe_04)
  

#Resumo dos MAEs
  print(" ")
  print(" ")
  print("RESUMO DOS MAEs:")
  print(paste("MAE do Modelo 01 LM: ",mean_error_p01))
  print(paste("MAE do Modelo 01 GD: ",mean_error_p01_gd))
  print(paste("MAE do Modelo 01 NE: ",mean_error_p01_ne))
  print(paste("MAE do Modelo 02 LM: ",mean_error_p02))
  print(paste("MAE do Modelo 02 GD: ",mean_error_p02_gd))
  print(paste("MAE do Modelo 02 NE: ",mean_error_p02_ne))
  print(paste("MAE do Modelo 03 LM: ",mean_error_p03))
  print(paste("MAE do Modelo 03 GD: ",mean_error_p03_gd))
  print(paste("MAE do Modelo 03 NE: ",mean_error_p03_ne))
  print(paste("MAE do Modelo 04 LM: ",mean_error_p04))
  print(paste("MAE do Modelo 04 GD: ",mean_error_p04_gd))
  print(paste("MAE do Modelo 04 NE: ",mean_error_p04_ne))
