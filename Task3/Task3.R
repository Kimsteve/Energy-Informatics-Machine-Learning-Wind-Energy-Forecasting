setwd("~/Desktop/WindEnergyForecasting-Assignment2/Task3")


#read CSV file and prepare data
Data <- read.csv("TrainData.csv", sep=",", header=TRUE)
WeatherForecastData <- read.csv("WeatherForecastInput.csv")
SolutionData <- read.csv("Solution.csv")

Data$TIMESTAMP <- strptime(Data$TIMESTAMP, "%Y%m%d %H:%M")
Data$TIMESTAMP2 <- as.numeric(Data$TIMESTAMP - Data$TIMESTAMP[1]) / 3600 / 24 # second to days (from 2012-01-01 01:00)

WeatherForecastData$TIMESTAMP <- strptime(WeatherForecastData$TIMESTAMP, "%Y%m%d %H:%M")
WeatherForecastData$TIMESTAMP2 <- as.numeric(WeatherForecastData$TIMESTAMP - Data$TIMESTAMP[1])  / 3600 / 24 # second to days (from 2012-01-01 01:00)

#function used for calculation RMSE metric.
rmse <- function(y, y_pred) sqrt(mean((y - y_pred)^2))


# build a model
model.lm <- lm(POWER ~ TIMESTAMP2, data = Data)
y_hat.lm <- predict(model.lm, WeatherForecastData)
rmse(SolutionData$POWER, y_hat.lm)
rmse(SolutionData$POWER, mean(Data$POWER)) # compare with null model

WeatherForecastData$TIMESTAMPdays <- WeatherForecastData$TIMESTAMP2 *3600*24 - 669
plot(SolutionData$POWER ~ WeatherForecastData$TIMESTAMPdays, type="l")
abline(model.lm, col = "red")
#model.lm$coefficients

write.csv(data.frame(TIMESTAMP = WeatherForecastData$TIMESTAMP, FORECAST = y_hat.lm), "ForecastTemplate3-LR.csv", row.names = FALSE)
