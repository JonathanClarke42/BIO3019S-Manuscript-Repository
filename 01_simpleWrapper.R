##' This file is the wrapper from which all model components are called. Detailed descriptions of these components are found in their respective files. 
##' @Requirements Data is downloaded from Github, i.e. internet connection is needed to run this.
##' @Dependencies Data Preprocessing/01_Data.Import.R, Stan/01_Streamflow.model.stan
##' @Output This file only produces the posterior samples and writes them into a local file.

####Optionals####
setwd("C:/Users/jonat/Desktop/BIO3019S/BIO3019S-Streamflow-Project") #Set working directory to a local file.
rm(list=ls());cat("\014")

####Import data####
if(file.exists("functions/00_dataImport.R")) source("functions/00_dataImport.R") #calls a script to download data from Github

data <- dataChoice(resolution = "daily",
                   timeframe = c("2023-05-01","2023-07-31"),
                   metadata = T)
write.csv(data, file = "23MJD.csv")
####Compile and run the Stan model####
library(rstan)

options(mc.cores = parallel::detectCores()) #parallelizes the the number of cores your computer has.

rstan_options(auto_write = T) #Avoids recompilation of unchanged Stan programs.
model <- stan_model('stan/04.2_Scaling.stan')

fit <- sampling(model,
                data = list(N=nrow(data), 
                            S = data[,2], 
                            R = data[,5],
                            wind = data[,15]),
                iter = 4000, chains = 4,
                cores = 4,
                seed = 12345)

library(shinystan)
launch_shinystan(readRDS("C:\\Users\\jonat\\Desktop\\BIO3019S\\BIO3019S-Streamflow-Project\\fits\\04.2_Scaling.rds"))
