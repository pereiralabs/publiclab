#Local outlier factor

#Calculating
#install.packages("DMwR")
library(DMwR)
df <- iris[, 1:4]
scores <- lofactor(df, k=5)
outliers <- order(scores, decreasing=T)[1:5]

#Plotting
xy <- prcomp(df)$x[, 1:2]
pch <- rep(".", nrow(xy))
pch[outliers] <- "+"
plot(xy, pch=pch)

install.packages("curl")
install.packages("libcurl")
install.packages("TTR")
install.packages("quantmod")
