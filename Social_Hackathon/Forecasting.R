library(data.table)
library(forecast)
library(tseries)
library(ggplot2)

setwd("~/My_Work/Social_Hackathon")
mydata = read.csv("train.csv", fill = TRUE, header = TRUE)
names(mydata)
mydata$timestamp = as.Date(mydata$timestamp, "%Y-%m-%d %H:%M:%S")
subset = mydata[mydata$building_number == 1,]
subset = subset[order(subset$timestamp),]
names(subset)
subset = subset[names(subset) %in% c("timestamp", "main_meter")]

subset = subset[subset$timestamp >= '2017-09-01 00:00:00',]

ggplot(subset, aes(timestamp, main_meter)) + geom_line()

ggplot(subset[9660:11712,], aes(timestamp, main_meter)) + geom_line() + scale_x_date('1 hour')

series = ts(subset$main_meter)
subset$series = series

train = subset$series[0:11615]
test = subset$series[11616:11712]

count_ma = ts(na.omit(train), frequency=96)
decomp = stl(count_ma, s.window="periodic")
deseasonal_cnt <- seasadj(decomp)
plot(decomp)

adf.test(count_ma, alternative = "stationary")

Acf(count_ma, main='')
Pacf(count_ma, main='')

auto.arima(deseasonal_cnt, seasonal=FALSE)

fit2 = arima(deseasonal_cnt, order=c(2,0,2), seasonal = list(order=c(0,0,0), period = 96))
fit2

fcast <- forecast(fit2, h=96)
plot(fcast)
fcast
