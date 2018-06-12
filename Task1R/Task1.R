#set the current working directory
setwd("~/Desktop/WindEnergyForecasting-Assignment2/Task1R")

#read CSV file and prepare 
Data <- read.csv("TrainData.csv", sep=",", header=TRUE)
WeatherForecastData <- read.csv("WeatherForecastInput.csv")
SolutionData <- read.csv("Solution.csv")

Data$TIMESTAMP <- strptime(Data$TIMESTAMP, "%Y%m%d %H:%M")
Data$TIMESTAMP2 <- as.numeric(Data$TIMESTAMP - Data$TIMESTAMP[1]) / 3600 / 24 # zamiana sekund na dni
# liczba dni od 1. daty
set.seed(1)
test <- sample(1:nrow(Data), 2000)
DataTrain <- Data[-test, ]; DataTest <- Data[test, ]

rmse <- function(y, y_pred) sqrt(mean((y - y_pred)^2))
rmse(SolutionData$POWER, mean(Data$POWER)) # so called "null model"


#scattplot points
plot(POWER ~ WS10, data = Data)

lmOut = lm(POWER ~ WS10, data = Data)
summary(lmOut)
rmse(SolutionData$POWER, predict(lmOut, WeatherForecastData))


#################################################
#kNN model
################################################
# install.packages("FNN")
library (FNN)

# Look at kNN regression 
knnmodel1 <- knn.reg(train = as.matrix(DataTrain$WS10), test = as.matrix(DataTest$WS10), y = DataTrain$POWER, k = 1)
rmse(DataTest$POWER, knnmodel1$pred)

k <- c(1:10, seq(12, 30, by = 2), seq(35, 50, by = 5))
results <- numeric(length(k))
for (i in seq_along(k))
  results[i] <-
  rmse(DataTest$POWER, 
       knn.reg(train = as.matrix(DataTrain$WS10), test = as.matrix(DataTest$WS10), y = DataTrain$POWER, k = k[i])$pred)

plot(k, results, type = "l")

k <- seq(100, 2000, by = 100)
results <- numeric(length(k))
for (i in seq_along(k))
  results[i] <-
  rmse(DataTest$POWER, 
       knn.reg(train = as.matrix(DataTrain$WS10), test = as.matrix(DataTest$WS10), y = DataTrain$POWER, k = k[i])$pred)

plot(k, results, type = "l")
# k = 1400
knnOut <- knn.reg(train = as.matrix(Data$WS10), test = as.matrix(WeatherForecastData$WS10), y = Data$POWER, k = 1400)

rmse(SolutionData$POWER, knnOut$pred)


## SVR

library(e1071)

eps <- seq(0.1, 0.4, by = 0.1)
results <- numeric(length(eps))
for (i in seq_along(k)) {
  SVROut <- svm(POWER ~ WS10, data = DataTrain, eps = eps[i])
  results[i] <- rmse(DataTest$POWER, predict(SVROut, DataTest))
  print(results[i])
}

SVROut <- svm(POWER ~ WS10, data = Data, eps = 0.3)
rmse(SolutionData$POWER, predict(SVROut, WeatherForecastData))


## NN

library(neuralnet)

NNOut <- neuralnet(POWER ~ WS10, data = Data, hidden = c(4, 2))
rmse(SolutionData$POWER, compute(NNOut, WeatherForecastData$WS10)$net.result)


LmOutPred <- predict(lmOut, WeatherForecastData)
write.csv(data.frame(TIMESTAMP = WeatherForecastData$TIMESTAMP, FORECAST = LmOutPred), "ForecastTemplate1-LR.csv", row.names = FALSE)

KNNOutPred <- knnOut$pred
write.csv(data.frame(TIMESTAMP = WeatherForecastData$TIMESTAMP, FORECAST = KNNOutPred ), "ForecastTemplate1-kNN.csv", row.names = FALSE)

SVRPred <- predict(SVROut, WeatherForecastData)
write.csv(data.frame(TIMESTAMP = WeatherForecastData$TIMESTAMP, FORECAST = SVRPred), "ForecastTemplate1-SVR.csv", row.names = FALSE)

NNPred <- compute(NNOut, WeatherForecastData$WS10)$net.result
write.csv(data.frame(TIMESTAMP = WeatherForecastData$TIMESTAMP, FORECAST = NNPred), "ForecastTemplate1-NN.csv", row.names = FALSE)


plot(SolutionData$POWER ~ WeatherForecastData$WS10)
lines(WeatherForecastData$WS10, LmOutPred, col = "red")

plot(SolutionData$POWER ~ WeatherForecastData$WS10)
lines(sort(WeatherForecastData$WS10), sort(KNNOutPred), col = "blue")

plot(SolutionData$POWER ~ WeatherForecastData$WS10)
lines(sort(WeatherForecastData$WS10), sort(SVRPred), col = "brown")

plot(SolutionData$POWER ~ WeatherForecastData$WS10)
lines(sort(WeatherForecastData$WS10), sort(NNPred), col = "violet")


#plot(SolutionData$POWER ~ WeatherForecastData$WS10)
#lines(WeatherForecastData$WS10, LmOutPred, col = "red")
#lines(sort(WeatherForecastData$WS10), sort(KNNOutPred), col = "blue")
#lines(sort(WeatherForecastData$WS10), sort(SVRPred), col = "brown")
#lines(sort(WeatherForecastData$WS10), sort(NNPred), col = "violet")


rmse(SolutionData$POWER, LmOutPred)
rmse(SolutionData$POWER, KNNOutPred)
rmse(SolutionData$POWER, SVRPred)
rmse(SolutionData$POWER, NNPred)
