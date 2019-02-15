#Clustering Algorithms

#K-Means
  iris2d=prcomp(iris[,-5])$x[,1:2]
  ii=iris2d[1:20,]
  dd=dist(ii)
  h.c=hclust(dd,"complete")
  plot(h.c,main="Complete")
  plot(h.c,main="Complete",hang=-1)
  
#Hierarchical Clustering
  h.s=hclust(dd,"single")
  plot(h.s,main="Single")
  h.a=hclust(dd,"ave")
  plot(h.a,main="Average")
  h.w=hclust(dd,"ward")
  plot(h.w,main="Ward")
  
#Hierarchical Clustering with Cutree
  plot(iris2d)
  dd=dist(iris2d)
  c.s=cutree(hclust(dd,"single"),k=5)
  plot(iris2d,col=c.s,main="Single")
  c.c=cutree(hclust(dd,"complete"),k=5)
  plot(iris2d,col=c.c,main="Complete")
  c.a=cutree(hclust(dd,"ave"),k=5)
  plot(iris2d,col=c.a,main="Average")
  c.w=cutree(hclust(dd,"ward.D"),k=5)
  plot(iris2d,col=c.w,main="ward")
  
#Variações do K-Means
  library(datasets)
  data(iris)
  library(flexclust)
  clmedians=kcca(iris[,1:4],k=3)
  library(cluster)
  clmedoid=pam(iris[,1:4],k=3)
  library(e1071)
  clcmeans=cmeans(iris[,1:4],3,m=2)
  
#Install de packages
  #install.packages("flexclust")
  library(flexclust)
  kcca(x = iris2d, k=5)
  
  #install.packages("cluster")
  library(cluster)
  pam(iris2d, k=5)
  
  #install.packages("e1071")
  library(e1071)
  cmeans(iris2d, 5)
  
#Variações de K-Means
  library(datasets)
  data(iris)
  library(flexclust)
  clmedians=kcca(iris[,1:4],k=3); clmedians
  library(cluster)
  clmedoid=pam(iris[,1:4],k=3); clmedoid
  library(e1071)
  clcmeans=cmeans(iris[,1:4],3,m=2); clcmeans
  
#DBSCAN
  #install.packages('dbscan')
  library('dbscan')
  k <- kNNdist(iris2d, k = 4)
  kNNdistplot(iris2d, k = 4)
  
  d <- dbscan(iris2d, 0.1, minPts = 4)
  plot(iris2d,col=(d$cluster+1),main="dbscan")