#####################################
#		MDC - Machine Learning		#
#				UNICAMP				#
#			Linear Regression		#
#####################################

#Importar dataset
  abalone_data <- read.csv("/home/felipe/Downloads/abalone.data", header = TRUE, sep = ",", stringsAsFactors = F) 

#Verificaçao de dataset
  n_elements <- dim(abalone_data)[1]
  n_features <- dim(abalone_data)[2]

#Divisão do dataset em treinamento e teste
  #Teste é 20% dos dados
  t <- trunc((n_elements/100)*20)
  abalone_data_test <- abalone_data[c(1:t),]
  
  #Dados de treinamento (80% dos dados)
  abalone_data_treinamento <- abalone_data[c(t:n_elements),]
  

#####################################
#############Modelo 01###############
#####################################
  
#Calculo de regressao linear com LM - Modelo 01
  r_lm_01 <- lm(formula=rings~length+diameter+height+whole_weight+shucked_weight+viscera_weight+shell_weight,data=abalone_data_treinamento)
  
#Predict - Modelo 01
  p_01 <- predict (r_lm_01,abalone_data_test)
  
#Cálculo de erro médio absoluto (MAE) - Modelo 01
  abalone_data_test$rings_prediction_01 <- p_01
  abalone_data_test$error_p01 <- abs(abalone_data_test$rings - abalone_data_test$rings_prediction_01)
  e_01 <- mean(abalone_data_test$error_p01)
  print(paste("Erro do modelo 01: ",e_01))
  
  
#####################################
#############Modelo 02###############
#####################################
  
  
#Calculo de regressao linear com LM - Modelo 02
  r_lm_02 <- lm(formula=rings~length+diameter+height+whole_weight,data=abalone_data_treinamento)
  
#Predict - Modelo 02
  p_02 <- predict (r_lm_02,abalone_data_test)
  
#Cálculo de erro médio absoluto (MAE) - Modelo 02
  abalone_data_test$rings_prediction_02 <- p_02
  abalone_data_test$error_p02 <- abs(abalone_data_test$rings - abalone_data_test$rings_prediction_02)
  e_02 <- mean(abalone_data_test$error_p02)
  print(paste("Erro do modelo 02: ",e_02))
  
  
#####################################
#############Modelo 03###############
#####################################
  
#Calculo de regressao linear com LM - Modelo 03
  r_lm_03 <- lm(formula=rings~length*diameter*height*whole_weight*shucked_weight*viscera_weight*shell_weight,data=abalone_data_treinamento)
  
#Predict - Modelo 03
  p_03 <- predict (r_lm_03,abalone_data_test)
  
#Cálculo de erro médio absoluto (MAE) - Modelo 03
  abalone_data_test$rings_prediction_03 <- p_03
  abalone_data_test$error_p03 <- abs(abalone_data_test$rings - abalone_data_test$rings_prediction_03)
  e_03 <- mean(abalone_data_test$error_p03)
  print(paste("Erro do modelo 03: ",e_03))
  
