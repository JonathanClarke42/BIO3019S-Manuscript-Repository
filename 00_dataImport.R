##' import data from the GitHub repository and produce appropriate subsets.
##' Import datasets loads the URLs to every dataset, but only downloads those presently used.
##' Subset datasets produces the subsets of the data used in the analysis.
##' @Dependencies this file is source by Stan.Wrapper, and the latter is sensitive to the variable names used in this file.

####Preprocessing####
#Load the URLs to the raw csv datasets in the GitHub repository.
url.daily <- "https://raw.githubusercontent.com/JonathanClarke42/streamflow_jonkershoek/refs/heads/main/data/data_daily_2025-07-09.csv"
url.hourly <- "https://raw.githubusercontent.com/JonathanClarke42/streamflow_jonkershoek/refs/heads/main/data/data_hourly_2025-07-09.csv"
url.meta.daily <- "https://raw.githubusercontent.com/JonathanClarke42/streamflow_jonkershoek/refs/heads/main/data/metadata_daily_2025-07-09.csv"
url.meta.hourly <- "https://raw.githubusercontent.com/JonathanClarke42/streamflow_jonkershoek/refs/heads/main/data/metadata_hourly_2025-07-09.csv"

#Construct the general daily dataset
dat_daily <- read.csv(url.daily)             #load the data from GitHub
dat_daily <- na.omit(dat_daily) #throw out unused fields.

dat_daily[,2] <- dat_daily[,2]*86400*1000                       #transform average streamflow in cumsecs to litres dailys, via formula number of cumsecs * seconds in a day * number of cumsecs * number of litres to a cubic meter.
#dat_daily[,5] <- dat_daily[,5]*2460000                          #transform milimeteres rainfall into total litres, via formula rainfall (in millimeters) * catchment area size (in metres square)

#Construct the general hourly dataset
dat_hourly <- read.csv(url.hourly)         #load the data from GitHub
dat_hourly <- na.omit(dat_hourly) #throw out unused fields.

dat_hourly[,1] <- substr(dat_hourly$Date,start = 1,stop = 10)     #remove the time element from each date.
dat_hourly[,2] <- dat_hourly[,2]*3600*1000                        #transform average streamflow in cumsecs to litres hourly, via formula number of cumsecs * seconds in an hour * number of cumsecs * number of litres to a cubic meter.
dat_hourly[,3] <- dat_hourly[,3]                                  #transform milimeteres rainfall into total litres, via formula rainfall (in millimeters) * catchment area size (in metres square)

####User choice function####
##'function to initialize the correct data object, according to specified choice of resolution and time frame.
##' @param resolution string, either "hourly" or "daily"
##' @param timeframe list of two strings, first element time frame start, second time frame end, in format "yyyy-mm-dd". Note all days and months are expressed with two digits, even if the first is a zero.
##' @param metadata boolean, whether or not to print to the console the metadata.

dataChoice <- function(resolution,timeframe,metadata){
  #selects the specified data set.
  if(resolution == "hourly"){
    data = dat_hourly
  } else {data = dat_daily}
  
  #retains only dates within the specified range.
  data <- data[c(match(timeframe[1],data[,1]):match(timeframe[2],data[,1])),]
  rownames(data) <- NULL
  
  #prints the metadata is asked for.
  if(metadata){
    if(resolution == "hourly"){print(read.csv(url.meta.hourly))}
    if(resolution == "daily"){print(read.csv(url.meta.daily))}
  }
  
  return(data)
}