#################################
# MDC - Machine Learning		#
# Create Toy Dataset			#
#################################

rm(list=ls())
set.seed(42)
n <- 150 # number of data points
p <- 2   # dimension

sigma <- 1  # variance of the distribution
meanpos <- 0 # centre of the distribution of positive examples
meanneg <- 3 # centre of the distribution of negative examples

npos <- round(n/2) # number of positive examples
nneg <- n-npos # number of negative examples

# Generate the positive and negative examples
xpos <- matrix(rnorm(npos*p,mean=meanpos,sd=sigma),npos,p)
xneg <- matrix(rnorm(nneg*p,mean=meanneg,sd=sigma),npos,p)
x <- rbind(xpos,xneg)


# Generate the labels
y <- matrix(c(rep(1,npos),rep(-1,nneg)))

# Visualize the data
plot(x,col=ifelse(y>0,1,2))
legend("topleft",c("Positive","Negative"),col=seq(2),pch=1,text.col=seq(2))

#Now we split the data into a training set (80%) and a test set (20%):
## Prepare a training and a test set ##
ntrain <- round(n*0.8) # number of training examples
tindex <- sample(n,ntrain) # indices of training samples

xtrain <- x[tindex,]
xtest <- x[-tindex,]

ytrain <- y[tindex]
ytest <- y[-tindex]

istrain=rep(0,n)
istrain[tindex]=1

# Visualize
plot(x,col=ifelse(y>0,1,2),pch=ifelse(istrain==1,1,2))
legend("topleft",c("Positive Train","Positive Test","Negative Train","Negative Test"),
    col=c(1,1,2,2),pch=c(1,2,1,2),text.col=c(1,1,2,2))





############################################


nclust <- 30 # number of points in each cluster
n <- 4*nclust # number of data points
p <- 2   # dimension
sigma <- 0.8  # variance of the distribution
meanpos <- 0 # centre of the distribution of positive examples
meanneg <- 3 # centre of the distribution of negative examples

# Generate the positive and negative examples
x1 <- rnorm(nclust,mean= meanpos,sd=sigma)
x2 <- rnorm(nclust,mean= meanpos,sd=sigma)
nlx = cbind(x1,x2)
x1 <- rnorm(nclust,mean= meanneg,sd=sigma)
x2 <- rnorm(nclust,mean= meanneg,sd=sigma)
nlx= rbind(nlx,cbind(x1,x2))
x1 <- rnorm(nclust,mean= meanpos,sd=sigma)
x2 <- rnorm(nclust,mean= meanneg,sd=sigma)
nlx= rbind(nlx,cbind(x1,x2))
x1 <- rnorm(nclust,mean= meanneg,sd=sigma)
x2 <- rnorm(nclust,mean= meanpos,sd=sigma)
nlx= rbind(nlx,cbind(x1,x2))
nly = c(rep(1,2*nclust),rep(-1,2*nclust))

#plot(nlx,col=ifelse(nly>0,1,2),pch=ifelse(nonLinear_y>0,1,2))
#legend("topleft",c('Positive','Negative'),col=c(1,2),pch=c(1,2),text.col=c(1,2))
#grid()


ntrain <- round(n*0.8) # number of training examples
tindex <- sample(n,ntrain) # indices of training samples

nlxTrain <- nlx[tindex,]
nlxTest <- nlx[-tindex,]

nlyTrain <- nly[tindex]
nlyTest <- nly[-tindex]






