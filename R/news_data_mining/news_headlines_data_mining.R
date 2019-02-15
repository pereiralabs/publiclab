#News' Headlines Clustering Algorithm (with K-Means & Others)


#Loading libraries
  #install.packages("cluster") -- FOR PAM FUNCTION
  #install.packages("NLP") -- FOR NGRAMS FUNCTION
  library(NLP)  
  library(cluster)
  library(flexclust)

#First things first, let's clear the cache
  rm(list=ls())
  print(paste("Iniciando algoritmo:",Sys.time()))

#Loading the datasets
  dsFeatures <- read.csv("/home/felipe/lab/lab/R/news_data_mining/features.csv", header = TRUE, sep = ",", stringsAsFactors = T)
  dsHeadlines <- read.csv("/home/felipe/lab/lab/R/news_data_mining/headlines.csv", header = TRUE, sep = ",", stringsAsFactors = T)
  
#dsFeatures has 2181 features
  names(dsFeatures)
  
#dsHeadlines has 2 features and news are out of publish date order
  names(dsHeadlines)
  head(dsHeadlines)
  
#Using 2016 DATA
  dsHeadlines <- dsHeadlines[c(which(dsHeadlines$publish_date >= '20160101' & dsHeadlines$publish_date < '20170101')),]
  dsFeatures <- dsFeatures[c(which(dsHeadlines$publish_date >= '20160101' & dsHeadlines$publish_date < '20170101')),]
  
#Calculating some sample indices
  #n <- dim(dsFeatures)[1] # Number of elements
  #perc <- 0.4  # Percentage - e.g.: 0.1 == 10%
  #indices <- sample(n, n*perc) #Percentage of number of elements
  
#USING SAMPLE DATA FOR NOW
  #dsHeadlinesSample <- dsHeadlines[indices,]
  #dsHeadlines <- dsHeadlinesSample
  #dsFeaturesSample <- dsFeatures[indices, ]
  #dsFeatures <- dsFeaturesSample

#Calculating PCA with scale
  #print(paste("Inicio do calculo do PCA SCALE:",Sys.time()))
  #featuresScale <- prcomp(dsFeatures,scale=T)
  #print(paste("Fim do calculo do PCA SCALE:",Sys.time()))
  
#Calculating PCA WITHOUT scale
  print(paste("Inicio do calculo do PCA NO SCALE:",Sys.time()))
  featuresNoScale <- prcomp(dsFeatures,scale=F)
  print(paste("Fim do calculo do PCA NO SCALE:",Sys.time()))
  
#Checking variance of both models
  options(max.print = .Machine$integer.max)
  #summary(featuresScale) # 85% = @PC1546 / 90% = @PC1712
  summary(featuresNoScale) # 85% = @PC831 / 90% = @PC984
  
#Checking the most representative components
  #head(featuresScale$x[,1:831])
  head(featuresNoScale$x[,1:831])
  
#Defining objects to be used in clustering algorithms & USING SAMPLE DATA
  pcaNoScale = featuresNoScale$x[,1:831]
  
  #Sample data
  #ii=pcaNoScale[1:100,]
  
  #Full data
  ii=pcaNoScale

#Calculating distance between points
  print(paste("Inicio do calculo do Dist:",Sys.time()))
  dd=dist(ii)
  print(paste("Fim do calculo do Dist:",Sys.time()))
  
#In search of the perfect K
i = 1
k_min <- 4
k_max <- 20
v_max_size <- k_max - k_min
k <- k_min
k_vector = c(i:v_max_size)
kmeans_error_vector = k_vector
kmeans_sil_vector = k_vector
kmedians_error_vector = k_vector
kmedians_sil_vector = k_vector
kmedoids_error_vector = k_vector
kmedoids_sil_vector = k_vector

while (k <= k_max){
  print(paste("CALCULANDOO K = ",k))
  
  #K-Means
    print(paste("Inicio do calculo do K-Means :",Sys.time()))
    h.c.kmeans =kmeans(ii,k)
    print(paste("Fim do calculo do K-Means :",Sys.time()))
  
  #K-Medians
    print(paste("Inicio do calculo do K-Medians :",Sys.time()))
    h.c.kmedians = kcca(as.matrix(ii),k,family=kccaFamily("kmedians"))
    print(paste("Fim do calculo do K-Medians :",Sys.time()))
    
  #K-Medoids
    print(paste("Inicio do calculo do K-Medoids :",Sys.time()))
    h.c.kmedoids = pam(ii,k)
    print(paste("Fim do calculo do K-Medoids :",Sys.time()))
    
  #Erros quadraticos (variar k de 4 a 20)
      #K-means
      h.c.kmeans$tot.withinss
      
      #K-Medians
      #mean(h.c.kmedians@cldist) -- ERRADO
      info(h.c.kmedians,"distsum")
      
      #K-Medoids
      #Reference: https://stackoverflow.com/questions/38306259/compute-within-sum-of-squares-from-pam-cluster-analysis-in-r
      mean(h.c.kmedoids$objective)
    
  #Silhueta (variar k de 4 a 20)
      #K-Means
      ksil_kmeans=silhouette(h.c.kmeans$cluster,dist=dd)
      mean(ksil_kmeans[,3])
      summary(ksil_kmeans)$avg.width
      
      #K-Medians
      ksil_kmedians = silhouette(h.c.kmedians@cluster,dist=dd)
      summary(ksil_kmedians)$avg.width
      
      #K-Medoids
      ksil_kmedoids=silhouette(h.c.kmedoids$clustering,dist=dd)
      mean(ksil_kmedoids[,3])
      summary(ksil_kmedoids)$avg.width
      h.c.kmedoids$silinfo$avg.width
      
  #Build array of Error & Silhoutte for each K for each algorithm
      k_vector[i] = k
      #K-Means
      kmeans_error_vector[i] = h.c.kmeans$tot.withinss
      kmeans_sil_vector[i] = summary(ksil_kmeans)$avg.width
      
      #K-Medians
      kmedians_error_vector[i] = info(h.c.kmedians,"distsum")
      kmedians_sil_vector[i] = summary(ksil_kmedians)$avg.width
      
      #K-Medoids
      kmedoids_error_vector[i] = mean(h.c.kmedoids$objective)
      kmedoids_sil_vector[i] = h.c.kmedoids$silinfo$avg.width
  
  #Aux
  k <- k +1 
  i = i + 1
}  

#Plot
  #K-Means Error
  plot(k_vector,kmeans_error_vector,main="K-Means Error",type="l")
  #K-Means Sil
  plot(k_vector,kmeans_sil_vector,main="K-Means Silhouette",type="l")
  
  #K-Medians Error
  plot(k_vector,kmedians_error_vector,main="K-Medians Error",type="l")
  #K-Medians Sil
  plot(k_vector,kmedians_sil_vector,main="K-Medians Silhouette",type="l")
  
  #K-Means Error
  plot(k_vector,kmedoids_error_vector,main="K-Medoids Error",type="l")
  #K-Means Sil
  plot(k_vector,kmedoids_sil_vector,main="K-Medoids Silhouette",type="l")
  
#K-Means most frequent bigrams
  #Melhor K do K-MEANS é 16, iniciando com a construção do vetor de bigramas
  print(paste("Inicio do calculo do melhor K-Means :",Sys.time()))
  h.c.kmeans =kmeans(ii,19)
  print(paste("Fim do calculo do melhor K-Means :",Sys.time()))
  
  #Aux Vars
  cl = 1
  bg_words_cl1 = c("")
  bg_words_cl2 = c("")
  bg_words_cl3 = c("")
  bg_words_cl4 = c("")
  bg_words_cl5 = c("")
  bg_words_cl6 = c("")
  bg_words_cl7 = c("")
  bg_words_cl8 = c("")
  bg_words_cl9 = c("")
  bg_words_cl10 = c("")
  bg_words_cl11 = c("")
  bg_words_cl12 = c("")
  bg_words_cl13 = c("")
  bg_words_cl14 = c("")
  bg_words_cl15 = c("")
  bg_words_cl16 = c("")
  bg_words_cl17 = c("")
  bg_words_cl18 = c("")
  bg_words_cl19 = c("")
  bg_words_cl20 = c("")
  
  #For each cluster, let's get the most frequent bigrams
  while (cl <= 19) {
    #For each record in the cluster
    for (registro in which(h.c.kmeans$cluster==cl)) {
      #For each headline in the cluster
      for (h in dsHeadlines[registro,]$headline_text) {
        w <- strsplit(h, " ", fixed=T)[[1]]
        n <- ngrams(w, 2)
        #For each bigram in the headline, put the bigram in the corresponding cluster vector
        for (bigram in n) {
          
          if (cl==1){
            bg_words_cl1 = c(bg_words_cl1,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==2){
            bg_words_cl2 = c(bg_words_cl2,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==3){
            bg_words_cl3 = c(bg_words_cl3,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==4){
            bg_words_cl4 = c(bg_words_cl4,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==5){
            bg_words_cl5 = c(bg_words_cl5,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==6){
            bg_words_cl6 = c(bg_words_cl6,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==7){
            bg_words_cl7 = c(bg_words_cl7,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==8){
            bg_words_cl8 = c(bg_words_cl8,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==9){
            bg_words_cl9 = c(bg_words_cl9,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==10){
            bg_words_cl10 = c(bg_words_cl10,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==11){
            bg_words_cl11 = c(bg_words_cl11,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==12){
            bg_words_cl12 = c(bg_words_cl12,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==13){
            bg_words_cl13 = c(bg_words_cl13,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==14){
            bg_words_cl14 = c(bg_words_cl14,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==15){
            bg_words_cl15 = c(bg_words_cl15,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==16){
            bg_words_cl16 = c(bg_words_cl16,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==17){
            bg_words_cl17 = c(bg_words_cl17,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==18){
            bg_words_cl18 = c(bg_words_cl18,paste(bigram[1]," ",bigram[2]))
          }
          if (cl==19){
            bg_words_cl19 = c(bg_words_cl19,paste(bigram[1]," ",bigram[2]))
          }
        }
      }
    }
    cl = cl + 1
  }
  
  #Most frequent bigrams
    #For Cluster 1
    print("Most frequent Bigrams for K-Means Cluster 1 (term/frequency):")
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[1]
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[2]
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[3]
    
    #For Cluster 2
    print("Most frequent Bigrams for K-Means Cluster 2 (term/frequency):")
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[1]
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[2]
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[3]
    
    #For Cluster 3
    print("Most frequent Bigrams for K-Means Cluster 3 (term/frequency):")
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[1]
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[2]
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[3]
    
    #For Cluster 4
    print("Most frequent Bigrams for K-Means Cluster 4 (term/frequency):")
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[1]
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[2]
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[3]
    
    #For Cluster 5
    print("Most frequent Bigrams for K-Means Cluster 5 (term/frequency):")
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[1]
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[2]
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[3]
    
    #For Cluster 6
    print("Most frequent Bigrams for K-Means Cluster 6 (term/frequency):")
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[1]
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[2]
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[3]
    
    #For Cluster 7
    print("Most frequent Bigrams for K-Means Cluster 7 (term/frequency):")
    rev(table(bg_words_cl7)[order(table(bg_words_cl7))])[1]
    rev(table(bg_words_cl7)[order(table(bg_words_cl7))])[2]
    rev(table(bg_words_cl7)[order(table(bg_words_cl7))])[3]
    
    #For Cluster 8
    print("Most frequent Bigrams for K-Means Cluster 8 (term/frequency):")
    rev(table(bg_words_cl8)[order(table(bg_words_cl8))])[1]
    rev(table(bg_words_cl8)[order(table(bg_words_cl8))])[2]
    rev(table(bg_words_cl8)[order(table(bg_words_cl8))])[3]
    
    #For Cluster 9
    print("Most frequent Bigrams for K-Means Cluster 9 (term/frequency):")
    rev(table(bg_words_cl9)[order(table(bg_words_cl9))])[1]
    rev(table(bg_words_cl9)[order(table(bg_words_cl9))])[2]
    rev(table(bg_words_cl9)[order(table(bg_words_cl9))])[3]
    
    #For Cluster 10
    print("Most frequent Bigrams for K-Means Cluster 10 (term/frequency):")
    rev(table(bg_words_cl10)[order(table(bg_words_cl10))])[1]
    rev(table(bg_words_cl10)[order(table(bg_words_cl10))])[2]
    rev(table(bg_words_cl10)[order(table(bg_words_cl10))])[3]
    
  
    #For Cluster 11
    print("Most frequent Bigrams for K-Means Cluster 11 (term/frequency):")
    rev(table(bg_words_cl11)[order(table(bg_words_cl11))])[1]
    rev(table(bg_words_cl11)[order(table(bg_words_cl11))])[2]
    rev(table(bg_words_cl11)[order(table(bg_words_cl11))])[3]
    
    #For Cluster 12
    print("Most frequent Bigrams for K-Means Cluster 12 (term/frequency):")
    rev(table(bg_words_cl12)[order(table(bg_words_cl12))])[1]
    rev(table(bg_words_cl12)[order(table(bg_words_cl12))])[2]
    rev(table(bg_words_cl12)[order(table(bg_words_cl12))])[3]
    
    #For Cluster 13
    print("Most frequent Bigrams for K-Means Cluster 13 (term/frequency):")
    rev(table(bg_words_cl13)[order(table(bg_words_cl13))])[1]
    rev(table(bg_words_cl13)[order(table(bg_words_cl13))])[2]
    rev(table(bg_words_cl13)[order(table(bg_words_cl13))])[3]
    
    #For Cluster 14
    print("Most frequent Bigrams for K-Means Cluster 14 (term/frequency):")
    rev(table(bg_words_cl14)[order(table(bg_words_cl14))])[1]
    rev(table(bg_words_cl14)[order(table(bg_words_cl14))])[2]
    rev(table(bg_words_cl14)[order(table(bg_words_cl14))])[3]
    
    #For Cluster 15
    print("Most frequent Bigrams for K-Means Cluster 15 (term/frequency):")
    rev(table(bg_words_cl15)[order(table(bg_words_cl15))])[1]
    rev(table(bg_words_cl15)[order(table(bg_words_cl15))])[2]
    rev(table(bg_words_cl15)[order(table(bg_words_cl15))])[3]
    
    #For Cluster 16
    print("Most frequent Bigrams for K-Means Cluster 16 (term/frequency):")
    rev(table(bg_words_cl16)[order(table(bg_words_cl16))])[1]
    rev(table(bg_words_cl16)[order(table(bg_words_cl16))])[2]
    rev(table(bg_words_cl16)[order(table(bg_words_cl16))])[3]
    
    #For Cluster 17
    print("Most frequent Bigrams for K-Means Cluster 16 (term/frequency):")
    rev(table(bg_words_cl17)[order(table(bg_words_cl17))])[1]
    rev(table(bg_words_cl17)[order(table(bg_words_cl17))])[2]
    rev(table(bg_words_cl17)[order(table(bg_words_cl17))])[3]
    
    #For Cluster 18
    print("Most frequent Bigrams for K-Means Cluster 16 (term/frequency):")
    rev(table(bg_words_cl18)[order(table(bg_words_cl18))])[1]
    rev(table(bg_words_cl18)[order(table(bg_words_cl18))])[2]
    rev(table(bg_words_cl18)[order(table(bg_words_cl18))])[3]
    
    #For Cluster 19
    print("Most frequent Bigrams for K-Means Cluster 16 (term/frequency):")
    rev(table(bg_words_cl19)[order(table(bg_words_cl19))])[1]
    rev(table(bg_words_cl19)[order(table(bg_words_cl19))])[2]
    rev(table(bg_words_cl19)[order(table(bg_words_cl19))])[3]
  
    
#K-Medians most frequent bigrams
    #Melhor K do K-MEDIANS é 7, iniciando com a construção do vetor de bigramas
    #K-Medians
    print(paste("Inicio do calculo do melhor K-Medians :",Sys.time()))
    h.c.kmedians = kcca(as.matrix(ii),7,family=kccaFamily("kmedians"))
    print(paste("Fim do calculo do melhor K-Medians :",Sys.time()))
    
    #To make our lifes easier, let's keep the name
    h.c.kmeans <- h.c.kmedians
    
    #Aux Vars
    cl = 1
    bg_words_cl1 = c("")
    bg_words_cl2 = c("")
    bg_words_cl3 = c("")
    bg_words_cl4 = c("")
    bg_words_cl5 = c("")
    bg_words_cl6 = c("")
    bg_words_cl7 = c("")
   
    #For each cluster, let's get the most frequent bigrams
    while (cl <= 7) {
      #For each record in the cluster
      for (registro in which(h.c.kmeans@cluster==cl)) {
        #For each headline in the cluster
        for (h in dsHeadlines[registro,]$headline_text) {
          w <- strsplit(h, " ", fixed=T)[[1]]
          n <- ngrams(w, 2)
          #For each bigram in the headline, put the bigram in the corresponding cluster vector
          for (bigram in n) {
            
            if (cl==1){
              bg_words_cl1 = c(bg_words_cl1,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==2){
              bg_words_cl2 = c(bg_words_cl2,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==3){
              bg_words_cl3 = c(bg_words_cl3,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==4){
              bg_words_cl4 = c(bg_words_cl4,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==5){
              bg_words_cl5 = c(bg_words_cl5,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==6){
              bg_words_cl6 = c(bg_words_cl6,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==7){
              bg_words_cl7 = c(bg_words_cl7,paste(bigram[1]," ",bigram[2]))
            }
          }
        }
      }
      cl = cl + 1
    }
    
    #Most frequent bigrams
    #For Cluster 1
    print("Most frequent Bigrams for K-Medians Cluster 1 (term/frequency):")
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[1]
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[2]
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[3]
    
    #For Cluster 2
    print("Most frequent Bigrams for K-Medians Cluster 2 (term/frequency):")
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[1]
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[2]
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[3]
    
    #For Cluster 3
    print("Most frequent Bigrams for K-Medians Cluster 3 (term/frequency):")
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[1]
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[2]
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[3]
    
    #For Cluster 4
    print("Most frequent Bigrams for K-Medians Cluster 4 (term/frequency):")
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[1]
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[2]
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[3]
    
    #For Cluster 5
    print("Most frequent Bigrams for K-Medians Cluster 5 (term/frequency):")
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[1]
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[2]
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[3]
    
    #For Cluster 6
    print("Most frequent Bigrams for K-Medians Cluster 6 (term/frequency):")
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[1]
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[2]
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[3]
    
    #For Cluster 7
    print("Most frequent Bigrams for K-Medians Cluster 7 (term/frequency):")
    rev(table(bg_words_cl7)[order(table(bg_words_cl7))])[1]
    rev(table(bg_words_cl7)[order(table(bg_words_cl7))])[2]
    rev(table(bg_words_cl7)[order(table(bg_words_cl7))])[3]
  
    
#K-Medoids most frequent bigrams
    #Melhor K do K-MEDOIDS é 4, iniciando com a construção do vetor de bigramas
    #K-Medoids
    print(paste("Inicio do calculo do melhor K-Medoids :",Sys.time()))
    h.c.kmedoids = pam(ii,6)
    print(paste("Fim do calculo do melhor K-Medoids :",Sys.time()))
    
    #To make our lifes easier, let's keep the name
    h.c.kmeans <- h.c.kmedoids
    
    #Aux Vars
    cl = 1
    bg_words_cl1 = c("")
    bg_words_cl2 = c("")
    bg_words_cl3 = c("")
    bg_words_cl4 = c("")
    
    #For each cluster, let's get the most frequent bigrams
    while (cl <= 6) {
      #For each record in the cluster
      for (registro in which(h.c.kmeans$clustering==cl)) {
        #For each headline in the cluster
        for (h in dsHeadlines[registro,]$headline_text) {
          w <- strsplit(h, " ", fixed=T)[[1]]
          n <- ngrams(w, 2)
          #For each bigram in the headline, put the bigram in the corresponding cluster vector
          for (bigram in n) {
            
            if (cl==1){
              bg_words_cl1 = c(bg_words_cl1,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==2){
              bg_words_cl2 = c(bg_words_cl2,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==3){
              bg_words_cl3 = c(bg_words_cl3,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==4){
              bg_words_cl4 = c(bg_words_cl4,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==5){
              bg_words_cl5 = c(bg_words_cl5,paste(bigram[1]," ",bigram[2]))
            }
            if (cl==6){
              bg_words_cl6 = c(bg_words_cl6,paste(bigram[1]," ",bigram[2]))
            }
          }
        }
      }
      cl = cl + 1
    }
    
    #Most frequent bigrams
    #For Cluster 1
    print("Most frequent Bigrams for K-Medoids Cluster 1 (term/frequency):")
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[1]
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[2]
    rev(table(bg_words_cl1)[order(table(bg_words_cl1))])[3]
    
    #For Cluster 2
    print("Most frequent Bigrams for K-Medoids Cluster 2 (term/frequency):")
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[1]
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[2]
    rev(table(bg_words_cl2)[order(table(bg_words_cl2))])[3]
    
    #For Cluster 3
    print("Most frequent Bigrams for K-Medoids Cluster 3 (term/frequency):")
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[1]
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[2]
    rev(table(bg_words_cl3)[order(table(bg_words_cl3))])[3]
    
    #For Cluster 4
    print("Most frequent Bigrams for K-Medoids Cluster 4 (term/frequency):")
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[1]
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[2]
    rev(table(bg_words_cl4)[order(table(bg_words_cl4))])[3]
  
    #For Cluster 5
    print("Most frequent Bigrams for K-Medoids Cluster 4 (term/frequency):")
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[1]
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[2]
    rev(table(bg_words_cl5)[order(table(bg_words_cl5))])[3]
    
    #For Cluster 6
    print("Most frequent Bigrams for K-Medoids Cluster 4 (term/frequency):")
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[1]
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[2]
    rev(table(bg_words_cl6)[order(table(bg_words_cl6))])[3]
  
  