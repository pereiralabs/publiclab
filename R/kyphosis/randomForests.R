#Wine Quality - Logistic Regression

#Clean cache
rm(list=ls())


#Import data
trainData <- read.csv("/home/felipe/Downloads/kyphosis_train.data", header = TRUE, sep = ",", stringsAsFactors = F) 
valData <- read.csv("/home/felipe/Downloads/kyphosis_val.data", header = TRUE, sep = ",", stringsAsFactors = F)

summary(trainData)
table(trainData$Kyphosis)

library(rpart)

treeModel = rpart(formula=Kyphosis ~ Age + Number + Start, data=trainData, method="class", parms = list(split="gini"))

plot(treeModel, uniform=TRUE)
text(treeModel, use.n=TRUE, all=TRUE, cex=.8)

#Print the table with complexity parameters
printcp(treeModel)

prediction = predict(treeModel, valData)
prediction = as.numeric(prediction[,"present"] >= 0.5)
prediction[prediction==0] = "absent"
prediction[prediction==1] = "present"

CM = as.matrix(table(Actual = valData$Kyphosis, Predicted = prediction))
TPR = CM[2,2] / (CM[2,2] + CM[2,1])
TNR = CM[1,1] / (CM[1,1] + CM[1,2])
ACCNorm = mean(c(TPR, TNR))
print(paste("ACC Norm: ",ACCNorm))

#install.packages("randomForest")
library("randomForest")

help(randomForest)


#Train RF model
trainData$Kyphosis <- as.factor(trainData$Kyphosis)
rfModel = randomForest(formula=Kyphosis~Age + Number + Start,data= trainData, ntree=15, mtry=2 )

#This is just to plot the error
layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(rfModel, log="y")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(rfModel$err.rate),col=1:4,cex=0.8,fill=1:4)


#Confusion Matrix
rfPrediction = predict(rfModel, valData) 
rfCM = as.matrix(table(Actual = valData$Kyphosis, Predicted = rfPrediction))
rfTPR = rfCM[2,2] / (rfCM[2,2] + rfCM[2,1])
rfTNR = rfCM[1,1] / (rfCM[1,1] + rfCM[1,2])
rfACCNorm = mean(c(rfTPR, rfTNR))
print(paste("RFACC Norm: ",rfACCNorm))