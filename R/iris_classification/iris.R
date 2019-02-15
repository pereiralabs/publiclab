#Script for iris (flower) classification

#Removing objects from cache
rm(list=ls())

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

#Loading library
library(datasets)

#Function to classify iris
iris_classifier <- function(dataframe){
  i <- 1
  rows <- dim(dataframe)[1]
  iris_class <- c(1:rows)
  
  for (element in dataframe[,5]) {
    
    if (element == "setosa"){
      iris_class[i] <- 1
    }
    else {
      if (element == "versicolor"){
        iris_class[i] <- 2  
      }
      else{
        if (element == "virginica"){
          iris_class[i] <- 3
        }
      }
    } 
     
    #Aux counter   
    i <- i + 1
  }
  
  return(iris_class)
}

#Classifying the flowes
iris_class <- iris_classifier(iris)

#Choosing iris dataset
data(iris)

#Summary of iris
summary(iris)

#PCA1 calculation
iris.pca1 <- prcomp(iris[,1:4],scale=T)

#Checking variance (line = Proportion of Variance/Cumulative Proportion)
summary(iris.pca1)

#Checking the most representative components
head(iris.pca1$x[,1:2])

#PCA2 calculation
iris.pca2 <- prcomp(iris[,1:4],scale=F)

#Checking variance (line = Proportion of Variance/Cumulative Proportion)
summary(iris.pca2)

#Checking the most representative components
head(iris.pca2$x[,1])

#PCA3 calculation
iris.pca3 <- princomp(iris[,1:4],cor=T)

#Checking variance (line = Proportion of Variance/Cumulative Proportion)
summary(iris.pca3)

#Checking the most representative components
head(iris.pca3$x[,1:2])

#Using GD to find the right classification
iris_matrix <- iris.pca1$x[,1:2]
iris_matrix <- cbind(iris_matrix,iris_class)
gd_01 <- GD(iris_matrix[,1:2], iris_matrix[,3], learningRate=0.001, nIter=1000, plotCost=FALSE)

#Predicting the value with GD
gd_p_01 <- predict_0615(iris_matrix[,1:2], gd_01$theta) 

#Error calculation
pred_error <- gd_p_01
i <- 1
for (prediction in ceiling(gd_p_01)){
  #If its right, then it's 0. If it's wrong, then it's 1.
  if (prediction == iris_matrix[i,3]) {
    pred_error[i] <- 0
  }
  else {
    pred_error[i] <- 1
  }
  
  #Aux
  i <- i + 1
}

#Accuracy calculation
iris_matrix <- cbind(iris_matrix,ceiling(gd_p_01))
iris_matrix <- cbind(iris_matrix,gd_p_01)
iris_matrix <- cbind(iris_matrix,pred_error)
error_rate <- sum(iris_matrix[,6])/dim(iris_matrix)[1]
accuracy <- 1 - error_rate