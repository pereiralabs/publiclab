#################################
# MDC - Machine Learning		#
# Support Vector Machines		#
#################################


# load the toy datasets
#source("create_dataset.r")

#Then import the data
trainData <- read.csv("/home/felipe/Downloads/occupancy_dataTrain.txt", header = TRUE, sep = ",", stringsAsFactors = T)
valData <- read.csv("/home/felipe/Downloads/occupancy_dataTest.txt", header = TRUE, sep = ",", stringsAsFactors = T)

# load the kernlab package
#install.packages("kernlab")
#install.packages("e1071")
library(kernlab)

xtrain <- trainData[,2:6]
ytrain <- trainData[,7]
ytrain[ytrain==0] = -1

xtest <- valData[,2:6]
ytest <- valData[,7]
ytest[ytest==0] = -1

######## TRAIN LINEAR-SVM ############

# train the SVM
svp <- ksvm(as.matrix(xtrain),ytrain,type="C-svc",kernel="vanilladot",C=100,scaled=c())
#Instead of xtrain and ytrain, we can use a formula
#see help(ksvm)

# General summary
svp

# Attributes that you can access
attributes(svp)

# For example, the support vectors
coef(svp)
alphaindex(svp)
b(svp)

# Use the built-in function to pretty-plot the classifier
plot(svp,data=xtrain)




######## PLOT DECISION BOUNDARY ############

# Define the range of the plot
# First column is plotted vertically
plotDecisionBoundary = function(svp, xtrain){
  yr <- c(min(xtrain[,1]), max(xtrain[,1]))
  # Second column is plotted horizontally
  xr <- c(min(xtrain[,2]), max(xtrain[,2]))
  
  # Plot the points of xtrain with different signs for positive/negative and SV/non SV
  plot(xr,yr,type='n')
  ymat <- ymatrix(svp)
  points(xtrain[-SVindex(svp),2], xtrain[-SVindex(svp),1], pch = ifelse(ymat[-SVindex(svp)] < 0, 2, 1))
  points(xtrain[SVindex(svp),2], xtrain[SVindex(svp),1], pch = ifelse(ymat[SVindex(svp)] < 0, 17, 16))
  
  # Extract w and b from the model	
  w <- colSums(coef(svp)[[1]] * xtrain[SVindex(svp),])
  b <- b(svp)
  
  # Draw the lines 
  abline(b/w[1],-w[2]/w[1])
  abline((b+1)/w[1],-w[2]/w[1],lty=2)
  abline((b-1)/w[1],-w[2]/w[1],lty=2)
  
  title(main=paste("C = ", toString(param(svp)$C)))
}


plotDecisionBoundary(svp, xtrain)



########## PREDICT WITH SVM #############
# Predict labels on test
ypred = predict(svp,xtest)
as.matrix(table(Actual = ytest, Predicted = ypred))

# Compute accuracy
sum(ypred==ytest)/length(ytest)
# Compute at the prediction scores
ypredscore = predict(svp,xtest,type="decision")
# Check that the predicted labels are the signs of the scores
table(ypredscore > 0,ypred)






######## CROSS-VALIDATION SVM ############
svp <- ksvm(xtrain,ytrain,type="C-svc",kernel="vanilladot",C=100,scaled=c(), cross=5)
cross(svp)



####### INFLUENCE OF C #############
CList = 10^seq(-2,4)

par(ask=T)

for (C in CList){
  svp <- ksvm(xtrain,ytrain,type="C-svc",kernel="vanilladot",C=C,scaled=c())
  
  plotDecisionBoundary(svp,xtrain)
}

par(ask=F)








