#Script for iris (flower) clustering

#Removing objects from cache
rm(list=ls())

#Loading library
library(datasets)

#Showing info
head(iris)
summary(iris)

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

#Clustering with K-Means
cl <- kmeans(iris[,1:4],3,nstart=200,iter.max = 10000)

#Iris Matrix
iris_matrix <- iris[,5]
iris_matrix <- cbind(iris_matrix,iris_class)

#Error calculation
pred_error <- cl$cluster
i <- 1
for (prediction in cl$cluster){
  #If its right, then it's 0. If it's wrong, then it's 1.
  if (prediction == iris_matrix[i,2]) {
    pred_error[i] <- 0
  }
  else {
    pred_error[i] <- 1
  }
  
  #Aux
  i <- i + 1
}

#Accuracy calculation - It's WRONG because it doesnt mean that KMEANS GROUP 1 is IRIS GROUP 1
#It could be that KMEANS GROUP 1 is IRIS GROUP 3
iris_matrix <- cbind(iris_matrix,cl$cluster)
iris_matrix <- cbind(iris_matrix,pred_error)
error_rate <- sum(iris_matrix[,4])/dim(iris_matrix)[1]
accuracy <- 1 - error_rate

#S Calculation
s1 <- (cl$betweenss-cl$withinss[1])/max(cl$withinss,cl$betweenss)
s2 <- (cl$betweenss-cl$withinss[2])/max(cl$withinss,cl$betweenss)
s3 <- (cl$betweenss-cl$withinss[3])/max(cl$withinss,cl$betweenss)