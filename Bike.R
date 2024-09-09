library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(GGally)

data <- vroom("train.csv")

##EDA 

glimpse(data)

plot_intro(data)
plot_correlation(data)
plot_bar(data)
plot_histogram(data)

#Plots
weather <- ggplot(data= data, aes(x=weather)) +
              geom_boxplot()
weather
temp <- ggplot(data= data, aes(x= temp, y= atemp))+ 
          geom_point() +
          geom_smooth()
temp

humidity <- ggplot(data = data, aes(x= humidity, y= count)) +
              geom_point() + 
              geom_smooth()
humidity

tempcount <- ggplot(data = data, aes(x= temp, y= count)) +
  geom_point() + 
  geom_smooth()
tempcount

(weather + temp) / (humidity + tempcount)
