#set the current working directory
setwd("~/Desktop/WindEnergyForecasting-Assignment2/Task2")

#read CSV file and prepare 
Data <- read.csv("TrainData.csv", sep=",", header=TRUE)
WeatherForecastData <- read.csv("WeatherForecastInput.csv")
SolutionData <- read.csv("Solution.csv")

Data$TIMESTAMP <- strptime(Data$TIMESTAMP, "%Y%m%d %H:%M")
Data$TIMESTAMP2 <- as.numeric(Data$TIMESTAMP - Data$TIMESTAMP[1]) / 3600 / 24 # second to days (from 2012-01-01 01:00)
SolutionData$TIMESTAMP <- strptime(SolutionData$TIMESTAMP, "%Y%m%d %H:%M")
SolutionData$TIMESTAMP2 <- as.numeric(SolutionData$TIMESTAMP - Data$TIMESTAMP[1]) / 3600 / 24 # second to days (from 2012-01-01 01:00)

# Calculate the wind direction based on the zonal component and meridional component
Data$direction <- atan2(Data$V10, Data$U10)*(180/pi)
WeatherForecastData$direction <- atan2(WeatherForecastData$V10, WeatherForecastData$U10)*(180/pi)
#plot(POWER ~ direction, Data)
#plot(WS10 ~ direction, Data)
hist(Data$direction)


# Calculate the wind direction based on the zonal component and meridional component
Data$direction <- atan2(Data$V10, Data$U10)*(180/pi)
plot(WS10 ~ direction, Data)



#function used for calculation RMSE metric.
rmse <- function(y, y_pred) sqrt(mean((y - y_pred)^2))


# Then, you build Multiple Linear Regression (MLR) model between wind power generation and two weather parameters
model.MLR.wd <- lm(POWER ~ WS10 + direction, data = Data)
summary(model.MLR.wd)

# Predict the wind power production for the whole month 11.2013
# based on the MLR model and weather forecasting data in the file WeatherForecastInput.csv
y_hat.wd <- predict(model.MLR.wd, WeatherForecastData)
rmse(SolutionData$POWER, y_hat.wd)

# Predicted wind power production is saved in the file ForecastTemplate2.csv
write.csv(data.frame(TIMESTAMP = WeatherForecastData$TIMESTAMP, FORECAST = y_hat.wd), "ForecastTemplate2.csv", row.names = FALSE)

# Compare the prediction accuracy with the linear regression where only wind speed is considered.
model.MLR.w <- lm(POWER ~ WS10, data = Data)
summary(model.MLR.w)

y_hat.w <- predict(model.MLR.w, WeatherForecastData)
rmse(SolutionData$POWER, y_hat.w)

#plot(SolutionData$POWER ~ WeatherForecastData$WS10)
#abline(model.MLR.simple, col = "red")




# Additionally, model directly based on the zonal and meridional component has been made.
model.MLR.wvu <- lm(POWER ~ WS10 + V10 + U10, data = Data)
summary(model.MLR.wvu)

y_hat.wvu <- predict(model.MLR.wvu, WeatherForecastData)
rmse(SolutionData$POWER, y_hat.wvu)


# cor(as.data.frame(Data)[,2:9])


# plot a time-series figure for 11.2013 which has three curves.
SolutionData$TIMESTAMPdays <- SolutionData$TIMESTAMP2*3600*24 - 669
# One curve shows the true wind energy measurement,
plot(POWER ~ TIMESTAMPdays, SolutionData, type = "line")
# the 2nd curve shows the wind power forecasts results by using linear regression, 
lines(SolutionData$TIMESTAMPdays, y_hat.w, col="blue" )
# and the 3rd curve shows the wind power forecasts results by using multiple linear regression
lines(SolutionData$TIMESTAMPdays, y_hat.wd, col = "red")



#------------------------------------------------------------------------------------------------------------------
# EXTRA TESTS
library(ggplot2)
ggplot(Data, aes(x = direction, y = POWER)) + geom_point() + geom_smooth()

Data$U10_25 <- Data$U10*cos(25 * pi/180)- Data$V10*sin(25 * pi/180)
Data$V10_25 <- Data$U10*sin(25 * pi/180) + Data$V10*cos(25 * pi/180)

Data$direction2 <- abs(atan2(Data$V10_25, Data$U10_25)*(180/pi))

# A rotation through angle θ [https://en.wikipedia.org/wiki/Rotation_matrix]
WeatherForecastData$U10_25 <- WeatherForecastData$U10*cos(25 * pi/180)- WeatherForecastData$V10*sin(25 * pi/180)
WeatherForecastData$V10_25 <- WeatherForecastData$U10*sin(25 * pi/180) + WeatherForecastData$V10*cos(25 * pi/180)
WeatherForecastData$direction2 <- abs(atan2(WeatherForecastData$V10_25, WeatherForecastData$U10_25)*(180/pi))
ggplot(Data, aes(x = direction, y = POWER)) + geom_point() + geom_smooth()

# Then, you build Multiple Linear Regression (MLR) model between wind power generation and two weather parameters
model.MLR.dir2 <- lm(POWER ~ WS10 + direction2, data = Data)
rmse(Data$POWER, predict(model.MLR.dir2))
summary(model.MLR.dir2)
cor.test(Data$POWER, Data$direction2)
ggplot(Data, aes(x = direction2, y = WS10)) + geom_point() + geom_smooth()
ggplot(Data, aes(x = direction, y = WS10)) + geom_point() + geom_smooth()

model.MLR.pred <- predict(model.MLR.dir2, WeatherForecastData)

# mean(Data$direction2[Data$WS10 > quantile(Data$WS10, 0.9)] < 90)
#> mean(Data$direction2[Data$WS10 > quantile(Data$WS10, 0.75)] < 90)
#> mean(Data$direction2[Data$WS10 > quantile(Data$WS10, 0.9)] < 90)
#> mean(Data$direction2[Data$WS10 > quantile(Data$WS10, 0.9)] < 90)

y_hat.dir2 <- predict(model.MLR.dir2, WeatherForecastData)
rmse(SolutionData$POWER, y_hat.dir2)

#------------------------------------------------------------------------------------------------------------------



