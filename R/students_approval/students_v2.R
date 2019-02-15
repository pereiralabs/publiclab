#Students Approval - Supervised Machine Learning Algorithm (Classification)

#Let's load the pre-reqs
  library("randomForest")
  library("rpart")

#Let's clean the cache
  rm(list=ls())

#Then import the data
  trainData <- read.csv("/home/felipe/Downloads/student_performance_train.data", header = TRUE, sep = ",", stringsAsFactors = T)
  valData <- read.csv("/home/felipe/Downloads/student_performance_test.data", header = TRUE, sep = ",", stringsAsFactors = T)

#Now let's check the data we've got
  #Let's get the big picture
  summary(trainData)
  #Schools are not balanced (GP=406/MS=166)
  table(trainData$school)
  #trainData <- rbind(trainData,trainData[trainData$school=="GP",])
  #trainData <- rbind(trainData,trainData[trainData$school=="GP",])
  #trainData <- rbind(trainData,trainData[trainData$school=="GP",])
  
  #Address is not balanced (but we kinda expected that I guess)
  table(trainData$address)
  #Famsize is not balanced and is surprising
  table(trainData$famsize)
  #Parent Status is not balanced (but we kinda expected that I guess)
  table(trainData$Pstatus)
  #Guardian is not balanced (but we kinda expected that I guess)
  table(trainData$guardian)
  
#Baseline Tree Model (Gini & Info)
  treeModelGini = rpart(formula=approved ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + absences, data=trainData, method="class", parms = list(split="gini"))
  treeModelInfo = rpart(formula=approved ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + absences, data=trainData, method="class", parms = list(split="information"))
  
#RandomForest - 01
  trainData$approved <- as.factor(trainData$approved)
  rfModel01 = randomForest(formula=approved ~ .,data= trainData, ntree=10, mtry=3 )
  
#RandomForest - 02
  trainData$approved <- as.factor(trainData$approved)
  rfModel02 = randomForest(formula=approved ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + absences,data= trainData, ntree=10, mtry=6 )
  
#RandomForest - 03
  trainData$approved <- as.factor(trainData$approved)
  rfModel03 = randomForest(formula=approved ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + absences,data= trainData, ntree=20, mtry=3 )

#RandomForest - 04
  trainData$approved <- as.factor(trainData$approved)
  rfModel04 = randomForest(formula=approved ~ school + sex + age + address + famsize + Pstatus + Medu + Fedu + Mjob + Fjob + reason + guardian + traveltime + studytime + failures + schoolsup + famsup + paid + activities + nursery + higher + internet + romantic + famrel + freetime + goout + Dalc + Walc + health + absences,data= trainData, ntree=20, mtry=6 )
  
#Predicting the models
  predTmGini = predict(treeModelGini, valData)
  predTmInfo = predict(treeModelInfo, valData)
  predRf01 = predict(rfModel01, valData) 
  predRf02 = predict(rfModel02, valData) 
  predRf03 = predict(rfModel03, valData) 
  predRf04 = predict(rfModel04, valData) 
  
#Calculating the norm
  #Func to calculate the norm values
  calcAccNorm <- function(data, predicted) {
    rfCM = as.matrix(table(Actual = data, Predicted = predicted))
    rfTPR = rfCM[2,2] / (rfCM[2,2] + rfCM[2,1])
    rfTNR = rfCM[1,1] / (rfCM[1,1] + rfCM[1,2])
    rfACCNorm = mean(c(rfTPR, rfTNR))
    
    print(rfCM)
    
    rfACCNorm
    
  }
  
  #Data transformation
  predTmGini = as.numeric(predTmGini[,"1"] >= 0.5)
  predTmGini[predTmGini==0] = "0"
  predTmGini[predTmGini==1] = "1"
  predTmInfo = as.numeric(predTmInfo[,"1"] >= 0.5)
  predTmInfo[predTmInfo==0] = "0"
  predTmInfo[predTmInfo==1] = "1"
  
  #Normalizing the values
  treeGiniNorm <- calcAccNorm(valData$approved,predTmGini)
  treeInfoNorm <- calcAccNorm(valData$approved,predTmInfo)
  rfAccNorm01 <- calcAccNorm(valData$approved,predRf01)
  rfAccNorm02 <- calcAccNorm(valData$approved,predRf02)
  rfAccNorm03 <- calcAccNorm(valData$approved,predRf03)
  rfAccNorm04 <- calcAccNorm(valData$approved,predRf04)
  
#Displaying the normalized accuracy
  print(paste("Tree Gini:",treeGiniNorm))
  print(paste("Tree Info:",treeInfoNorm))
  print(paste("Random Forest 01:",rfAccNorm01))
  print(paste("Random Forest 02:",rfAccNorm02))
  print(paste("Random Forest 03:",rfAccNorm03))
  print(paste("Random Forest 04:",rfAccNorm04))