#################################
# MDC - Machine Learning		#
# Support Vector Machines		#
#################################


# load the toy datasets
source("create_dataset.r")

# load the kernlab package
#install.packages("kernlab")
#install.packages("e1071")
library(kernlab)

######## TRAIN LINEAR-SVM ############

# train the SVM
svp <- ksvm(xtrain,ytrain,type="C-svc",kernel="vanilladot",C=100,scaled=c())
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











########## Non-linear SVM #################
plot(nlx,col=ifelse(nly>0,1,2),pch=ifelse(nly>0,1,2))
legend("topleft",c('Positive','Negative'),col=c(1,2),pch=c(1,2),text.col=c(1,2))
grid()


#let's use a linear SVM
par(ask=T)

errlin <- numeric(length(CList))
for (i in seq(length(CList))) {
	svp <- ksvm(nlx,nly,type="C-svc",kernel="vanilladot",C=CList[i],scaled=c(), cross=5)

	plotDecisionBoundary(svp,nlx)
	errlin[i] <- cross(svp)
}


# Plot the CV error as a function of C
plot(CList,errlin,type='l',log="x",ylim=c(0,1),xlab="C",ylab="Error rate")
grid()







# Now with a RBF kernel and default parameters
svp <- ksvm(nlx,nly,type="C-svc",kernel='rbf')
plot(svp,data=nlx)


# altering C, but with automatic choice of sigma
errrbf <- numeric(length(CList))
for (i in seq(length(CList))) {
	svp <- ksvm(nlx,nly,type="C-svc",kernel='rbf', C=CList[i], cross=5)

	plot(svp,data=nlx)
	errrbf[i] <- cross(svp)
}

plot(CList,errlin,type='l',log="x",ylim=c(0,1),xlab="C",ylab="Error rate",col=1,lwd=2)
lines(CList,errrbf,col=2,lwd=2)
grid()
legend("topright",c('Linear','RBF Gaussian'),lwd=2,col=c(1,2))





### Altering C and Sigma
nc <- length(CList)
sigmaList <- 2^seq(-6,6)
nsigma <- length(sigmaList)
err <- matrix(0,nrow=nc,ncol=nsigma)
for (i in seq(nc)) {
	C <- CList[i]
	cat('C=',C)
	for (j in seq(nsigma)) {
		cat('.')
		svp <- ksvm(nlx,nly,type="C-svc",kernel='rbf',kpar=list(sigma=sigmaList[j]),C=C,cross=3)
		err[i,j] <- cross(svp)
	}
	cat('\n')
}
# Visualize
library(lattice)
dimnames(err) <- list(C=CList,sigma=sigmaList)
levelplot(err,scales=list(x=list(rot=90)),xlab="C",ylab=expression(sigma),main="Error rate")












