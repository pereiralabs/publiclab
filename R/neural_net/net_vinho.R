#################################
# MDC - Machine Learning		#
# Neural Networks				#
#################################


# load the toy datasets
#source("create_dataset.r")

# install and load nn package
#install.packages("neuralnet")
library(neuralnet)

#Then import the data
trainData <- read.csv("/home/felipe/Downloads/wineQuality_train.data", header = TRUE, sep = ",", stringsAsFactors = T)
valData <- read.csv("/home/felipe/Downloads/wineQuality_val.data", header = TRUE, sep = ",", stringsAsFactors = T)

# load the kernlab package
#install.packages("kernlab")
#install.packages("e1071")
library(kernlab)

xtrain <- trainData[,1:11]
ytrain <- trainData[,12]
ytrain[ytrain==0] = -1

xtest <- valData[,1:11]
ytest <- valData[,12]
ytest[ytest==0] = -1

#NN only work with formula, so we need to cbind xtrain and ytrain
#trainData = cbind(xtrain, ytrain)
#colnames(trainData)= c("temp", "hum", "light","co2","humratio","occupancy")

#trainData[1:3,]


# train a neuralnet
nnModel = neuralnet(formula="quality ~ alcohol + residual.sugar + chlorides + pH + sulphates", data=trainData, hidden=c(5,3), linear.output=FALSE) 
#linear.output = TRUE --> regression
#linear.output = FALSE --> classification (apply 'logistic' activation as default)


# General summary
summary(nnModel)

# Plot the network
plot(nnModel)


#### Let's predict our test set
nnCompute = compute(nnModel, xtest)
nnCompute

prediction = nnCompute$net.result

prediction[prediction < 0.5] = -1
prediction[prediction >= 0.5] = 1

CM = as.matrix(table(Actual = ytest, Predicted = prediction))
TPR = CM[2,2] / (CM[2,2] + CM[2,1])
TNR = CM[1,1] / (CM[1,1] + CM[1,2])
ACCNorm = mean(c(TPR, TNR))


