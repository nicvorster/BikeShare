library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(GGally)

data <- vroom("train.csv")
testData  <- vroom("test.csv")

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

### LINEAR REGRESSION MODEL
library(tidymodels)

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
  set_engine("lm") %>% # Engine = What R function to use
  set_mode("regression") %>% # Regression just means quantitative response
  fit(formula=count~ temp +weather + humidity + windspeed, data=data)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=testData) # Use fit to predict11
bike_predictions ## Look at the output

## Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_predictions %>%
bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
