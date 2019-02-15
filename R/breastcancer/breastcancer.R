#Breast Cancer Classification Problem

#Clean cache
rm(list=ls())

#Load datasets
brestCancer_testSet <- read.csv("/home/felipe/Downloads/breastCancer_test.data", header = TRUE, sep = ",", stringsAsFactors = F) 
brestCancer_trainSet <- read.csv("/home/felipe/Downloads/breastCancer_train.data", header = TRUE, sep = ",", stringsAsFactors = F) 

#Checking Classes
table(brestCancer_trainSet$class)
nrow(brestCancer_trainSet[brestCancer_trainSet[,11]==2,])
nrow(brestCancer_trainSet[brestCancer_trainSet[,11]==4,])

table(brestCancer_testSet$class)
nrow(brestCancer_testSet[brestCancer_trainSet[,11]==2,])
nrow(brestCancer_testSet[brestCancer_trainSet[,11]==4,])

#Removing the ID column
brestCancer_trainSet[,"id"]=NULL
brestCancer_testSet[,"id"]=NULL

########### Transform to numeric
brestCancer_trainSet$bare.nuclei = as.numeric(brestCancer_trainSet$bare.nuclei)
brestCancer_testSet$bare.nuclei = as.numeric(brestCancer_testSet$bare.nuclei)

#Removing NAs
brestCancer_trainSet <-  brestCancer_trainSet[!is.na(brestCancer_trainSet$bare.nuclei),] 
brestCancer_testSet <-  brestCancer_testSet[!is.na(brestCancer_testSet$bare.nuclei),] 

#Setting the flag
brestCancer_trainSet[brestCancer_trainSet$class == 2,10] <- 0
brestCancer_trainSet[brestCancer_trainSet$class == 4,10] <- 1

brestCancer_testSet[brestCancer_testSet$class == 2,10] <- 0
brestCancer_testSet[brestCancer_testSet$class == 4,10] <- 1

#GLM calculation
formula <- as.formula("class ~ clump.thickness + unif.cell.size + unif.cell.shape + marginal.adhesion + epithelial.cell.size + bare.nuclei  + bland.chromatin + normal.nucleoli + mitoses")
model = glm(formula, brestCancer_trainSet, family=binomial(link="logit"))

#Prediction
#p_01 <- predict (model,brestCancer_testSet, type="response")
testPred = predict(model, brestCancer_testSet[,1:9], type="response")
testPred[testPred >= 0.5] = 1
testPred[testPred < 0.5] = 0

#Confusion Matrix
cm = as.matrix((table(Actual = brestCancer_testSet$class, Predicted= testPred)))

#Accuracy
accuracy = sum(diag(cm))/sum(cm)
error = 1-accuracy
print(paste("Non-Normalized Accuracy",accuracy))

#Normalized Accuracy (because groups are not balanced)
  #Accuracy for negatives (class==0)
  #True Negative Rate
  tnr = cm[1,1] / sum(cm[1,])
  print(paste("True Negative Rate",tnr))
  
  #Accuracy for positives (class==1)
  #True Positive Rate
  tpr = cm[2,2] / sum(cm[2,])
  print(paste("True Positive Rate",tpr))
  
  #Normalized Accuracy
  normacc <- (tnr+tpr)/2
  print(paste("Normalized Accuracy",normacc))