#Nome: Felipe Pereira Tome dos Santos

#Pre-reqs
library(ggplot2)
rm(list = ls())

#Criação de dataframe com dados do cepagri/zanoni
  colunas <- c("str_datahora", "temperatura", "vento", "umidade", "sensacao")
  cepagridf <- read.csv("/home/felipe/Downloads/cepagri.csv", header = FALSE, sep = ";", col.names = colunas, stringsAsFactors = F) 

#Verificando a quantidade de linhas
  cepagridf_linhas <- dim(cepagridf)[1]
  print(paste("Quantidade de linhas no arquivo original: ", cepagridf_linhas))

#Filtrar data frame apenas para o período da análise (01/01/2015 - 31/12/2017)
  #Transformar data string em data posix_lt
  datahora <- as.POSIXlt(cepagridf$str_datahora,format = "%d/%m/%Y-%H:%M") 
  cepagridf <- cbind(cepagridf,datahora)
  
  #Filtro de datas
  cepagridf <- cepagridf[cepagridf$datahora >= "2015-01-01" & cepagridf$datahora < "2018-01-01",]

#Verificando ausência de dados
  sem_data <- sum(is.na(cepagridf[,1]))
  sem_temperatura <- sum(is.na(cepagridf[,2]))
  sem_vento <- sum(is.na(cepagridf[,3]))
  sem_umidade <- sum(is.na(cepagridf[,4]))
  sem_sensacao <- sum(is.na(cepagridf[,5]))
  print(paste("Quantidade de linhas sem data: ",sem_data))
  print(paste("Quantidade de linhas sem temperatura: ",sem_temperatura))
  print(paste("Quantidade de linhas sem vento: ",sem_vento))
  print(paste("Quantidade de linhas sem umidade: ",sem_umidade))
  print(paste("Quantidade de linhas sem sensação térmica: ",sem_sensacao))

#Verificando discrepância de dados
  print("Summary de Temperatura:")
  print(summary(cepagridf[,2]))
  print("Summary de Vento:")
  print(summary(cepagridf[,3]))
  print("Summary de Umidade:")
  print(summary(cepagridf[,4]))
  print("Summary de Sensação Térmica:")
  print(summary(cepagridf[,5]))

#Problemas encontrados
  #Retirar NAs
  cepagridf <- cepagridf[!is.na(cepagridf$sensacao), ]
  
  #Temperatura como caracter
    cepagridf[,2] <- as.character(cepagridf[,2])
    cepagridf[,2] <- as.numeric(cepagridf[,2])
    print("Summary de Temperatura corrigida:")
    print(summary(cepagridf[,2]))
  
  #Sensação Térmica de 99.9
    cepagridf <- cepagridf[cepagridf$sensacao != 99.9, ]
    print("Summary de Sensaçao Termica corrigida:")
    print(summary(cepagridf[,5]))
  
#Verificar dias com medições repetidas, o que apresenta algum problema, pois a temperatura não pode ser a mesma por 24hrs
  #Implementação de consecutive disponibilizada em classe
     consecutive <- function(vector , k = 1) {
       n <- length(vector)
       result <- logical(n)
       for (i in (1+k):n)
         if (all(vector[(i-k):(i-1)] == vector[i]))
           result[i] <- TRUE
       for (i in 1:(n-k))
         if (all(vector[(i+1):(i+k)] == vector[i]))
           result[i] <- TRUE
       return(result)
      }
    
  #Verificação de quantas vezes as temperaturas se repetem por 24h (isto indica erro)
    leituras_com_problema_temperatura <-  sum(consecutive(cepagridf$temperatura, 144))
    print(paste("Quantidade de leituras em que os mesmos registros de temperatura se repetiram por 24h (indicação de problema na coleta ou gravação dos dados): ",leituras_com_problema_temperatura))


#Criação de colunas necessárias para os agrupamentos
    cepagridf$ano <- format(cepagridf$datahora,format="%Y")
    cepagridf$mes <- format(cepagridf$datahora,format="%m")
    cepagridf$dia <- format(cepagridf$datahora,format="%d")
    cepagridf$anomes <- format(cepagridf$datahora,format="%Y%m")
    cepagridf2015 <- cepagridf[cepagridf$datahora >= "2015-01-01" & cepagridf$datahora < "2016-01-01",]
    cepagridf2016 <- cepagridf[cepagridf$datahora >= "2016-01-01" & cepagridf$datahora < "2017-01-01",]
    cepagridf2017 <- cepagridf[cepagridf$datahora >= "2017-01-01" & cepagridf$datahora < "2018-01-01",]
    
#Plot  temp minima pra cada mes por 3 anos 
    temp_min_por_mes <- aggregate(cepagridf$temperatura , list(cepagridf$anomes), min)
    colnames(temp_min_por_mes) <- c("anomes", "Temperatura_Minima")
    temp_min_por_mes$ano <- substr(temp_min_por_mes$anomes,1,4)
    temp_min_por_mes$mes <- substr(temp_min_por_mes$anomes,5,6)
    graph_temp_min <- ggplot(temp_min_por_mes, aes(x=mes,y=Temperatura_Minima,group=ano,colour=ano))
    graph_temp_min <- graph_temp_min + geom_line()
      
#Plot  temp maxima pra cada mes por 3 anos 
    temp_max_por_mes <- aggregate(cepagridf$temperatura , list(cepagridf$anomes), max)
    colnames(temp_max_por_mes) <- c("anomes", "Temperatura_Maxima")
    temp_max_por_mes$ano <- substr(temp_max_por_mes$anomes,1,4)
    temp_max_por_mes$mes <- substr(temp_max_por_mes$anomes,5,6)
    graph_temp_max <- ggplot(temp_max_por_mes, aes(x=mes,y=Temperatura_Maxima,group=ano,colour=ano))
    graph_temp_max <- graph_temp_max + geom_line()

#BoxPlot  da temperatura mensal (1 grafico por ano)
    graph_temp_box_2015 <- ggplot(cepagridf2015, aes(x=anomes,y=temperatura,group=anomes))
    graph_temp_box_2015 <- graph_temp_box_2015 + geom_boxplot()
    
    graph_temp_box_2016 <- ggplot(cepagridf2016, aes(x=anomes,y=temperatura,group=anomes))
    graph_temp_box_2016 <- graph_temp_box_2016 + geom_boxplot()
    
    graph_temp_box_2017 <- ggplot(cepagridf2017, aes(x=anomes,y=temperatura,group=anomes))
    graph_temp_box_2017 <- graph_temp_box_2017 + geom_boxplot()

#Plot umidade 
    umidade_avg_por_mes <- aggregate(cepagridf$umidade , list(cepagridf$anomes), mean)
    colnames(umidade_avg_por_mes) <- c("anomes", "Umidade_Media")
    umidade_avg_por_mes$ano <- substr(umidade_avg_por_mes$anomes,1,4)
    umidade_avg_por_mes$mes <- substr(umidade_avg_por_mes$anomes,5,6)
    graph_umidade <- ggplot(umidade_avg_por_mes, aes(x=mes,y=Umidade_Media,group=ano,colour=ano))
    graph_umidade <- graph_umidade + geom_point() + geom_line()

#Plot relacao ventoxtemperatura 
    vento_aux <- aggregate(cepagridf$vento , list(cepagridf$anomes), mean)
    colnames(vento_aux) <- c("Mes", "Vento_Medio")
    temp_aux <- aggregate(cepagridf$temperatura , list(cepagridf$anomes), mean)
    colnames(temp_aux) <- c("Mes", "Temperatura_Media")
    vento_temp <- vento_aux
    vento_temp <- cbind(vento_temp,temp_aux$Temperatura_Media)
    colnames(vento_temp) <- c("Mes", "Vento_Medio","Temperatura_Media")
    
    graph_vento_temp <- ggplot(vento_temp, aes(x=Mes,group=1))
    graph_vento_temp <- graph_vento_temp + geom_bar(stat="identity" , aes(y=Temperatura_Media, colour="Temperatura_Media"))
    graph_vento_temp <- graph_vento_temp + geom_line(aes(y=Vento_Medio, colour="Vento_Medio"))
    
    
#Plot relacao ventoxsensacao 
    vento_aux <- aggregate(cepagridf$vento , list(cepagridf$anomes), mean)
    colnames(vento_aux) <- c("Mes", "Vento_Medio")
    sensacao_aux <- aggregate(cepagridf$sensacao , list(cepagridf$anomes), mean)
    colnames(sensacao_aux) <- c("Mes", "Sensacao_Media")
    vento_sensacao <- vento_aux
    vento_sensacao <- cbind(vento_sensacao,sensacao_aux$Sensacao_Media)
    colnames(vento_sensacao) <- c("Mes", "Vento_Medio","Sensacao_Media")
    
    graph_vento_sensacao <- ggplot(vento_sensacao, aes(x=Mes,group=1))
    graph_vento_sensacao <- graph_vento_sensacao + geom_bar(stat="identity" , aes(y=Sensacao_Media, colour="Sensacao_Media"))
    graph_vento_sensacao <- graph_vento_sensacao + geom_line(aes(y=Vento_Medio, colour="Vento_Medio"))
    
  
#Plot relacao umidadextemperatura 
    umidade_aux <- aggregate(cepagridf$umidade , list(cepagridf$anomes), mean)
    colnames(umidade_aux) <- c("Mes", "Umidade_Media")
    temp_aux <- aggregate(cepagridf$temperatura , list(cepagridf$anomes), mean)
    colnames(temp_aux) <- c("Mes", "Temperatura_Media")
    umidade_temp <- umidade_aux
    umidade_temp <- cbind(umidade_temp,temp_aux$Temperatura_Media)
    colnames(umidade_temp) <- c("Mes", "Umidade_Media","Temperatura_Media")
    
    graph_umidade_temp <- ggplot(umidade_temp, aes(x=Mes,group=1))
    graph_umidade_temp <- graph_umidade_temp + geom_line(aes(y=Umidade_Media, colour="Umidade_Media"))
    graph_umidade_temp <- graph_umidade_temp + geom_line(aes(y=Temperatura_Media, colour="Temperatura_Media"))
    

#Plot de sensacao e temperatura 
    sensacao_aux <- aggregate(cepagridf$sensacao , list(cepagridf$anomes), mean)
    colnames(sensacao_aux) <- c("Mes", "Sensacao_Media")
    temp_aux <- aggregate(cepagridf$temperatura , list(cepagridf$anomes), mean)
    colnames(temp_aux) <- c("Mes", "Temperatura_Media")
    sensacao_temp <- sensacao_aux
    sensacao_temp <- cbind(sensacao_temp,temp_aux$Temperatura_Media)
    colnames(sensacao_temp) <- c("Mes", "Sensacao_Media","Temperatura_Media")
    
    graph_sensacao_temp <- ggplot(sensacao_temp, aes(x=Mes,group=1))
    graph_sensacao_temp <- graph_sensacao_temp + geom_line(aes(y=Sensacao_Media, colour="Sensacao_Media"))
    graph_sensacao_temp <- graph_sensacao_temp + geom_line(aes(y=Temperatura_Media, colour="Temperatura_Media"))
  
#Plot amplitude termica nos ultimos 3 anos 
    temp_maxmin_por_mes <- temp_min_por_mes
    temp_maxmin_por_mes$Temperatura_Maxima <- temp_max_por_mes$Temperatura_Maxima
    temp_maxmin_por_mes$amplitude <- temp_maxmin_por_mes$Temperatura_Maxima - temp_maxmin_por_mes$Temperatura_Minima
    temp_maxmin_por_mes$ano <- substr(temp_maxmin_por_mes$anomes,1,4)
    temp_maxmin_por_mes$mes <- substr(temp_maxmin_por_mes$anomes,5,6)
  
    
    graph_amplitude <- ggplot(temp_maxmin_por_mes, aes(x=mes,y=amplitude,group=ano,colour=ano))
    graph_amplitude <- graph_amplitude + geom_line() + geom_point()
    


