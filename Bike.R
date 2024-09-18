library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(GGally)
library(glmnet)
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


### POISSON MODEL

library(poissonreg)

data$weather <- as.factor(data$weather)
testData$weather <- as.factor(testData$weather)

pois_model <- poisson_reg() %>% #Type of model
  set_engine("glm") %>% # GLM = generalized linear model
  set_mode("regression") %>%
fit(formula=count~ temp + humidity + windspeed + weather, data=data)

## Generate Predictions Using Linear Model
bike_predictions <- predict(pois_model,
                            new_data=testData) # Use fit to predict10
bike_predictions ## Look at the output

## Format the Predictions for Submission to Kaggle
kaggle_submission2 <- bike_predictions %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission2, file="./PoisPreds.csv", delim=",")






### CLEANING DATA 
cleandata <- data %>% 
  select(-casual , -registered) %>% 
  mutate(count = log(count))
view(cleandata)
### Recipe
bike_recipe <- recipe(count ~ ., data = cleandata) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3, labels = c("Dry", "Rainy", "Snowwy"))) %>% 
  step_mutate(windspeed = ifelse(windspeed > 20, "Very Windy", ifelse(windspeed >10, "Slightly Windy", "Not Windy"))) %>%
  step_mutate(windspeed = factor(windspeed)) %>% 
  step_time(datetime, features=("hour")) %>% 
  step_mutate(datetime_hour = factor(datetime_hour)) %>% 
  step_mutate(season = factor(season)) %>% 
  step_rm(datetime) %>% 
  step_mutate(newtemp = (temp + atemp)/2) %>% 
  step_rm(temp, atemp)
prepped_recipe <- prep(bike_recipe)
baked <- bake(prepped_recipe, new_data = testData)


## Define a model
lin_model <- linear_reg() %>%
set_engine("lm") %>%
set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() %>%
add_recipe(bike_recipe) %>%
add_model(lin_model) %>%
fit(data=cleandata)

## Run all the steps on test data
lin_preds <- predict(bike_workflow, new_data = testData)
lin_preds <- exp(lin_preds)

## Kaggle Submission 3
kaggle_submission3 <- lin_preds %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission3, file="./LinPreds.csv", delim=",")






#### PENALIZED LINEAR REGRESSION ###
penalized_recipe <- recipe(count ~ ., data = cleandata) %>% 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  step_mutate(weather = factor(weather, levels = 1:3, labels = c("Dry", "Rainy", "Snowwy"))) %>% 
  step_mutate(windspeed = ifelse(windspeed > 20, "Very Windy", ifelse(windspeed >10, "Slightly Windy", "Not Windy"))) %>%
  step_mutate(windspeed = factor(windspeed)) %>% 
  step_time(datetime, features=("hour")) %>% 
  step_mutate(datetime_hour = factor(datetime_hour)) %>% 
  step_mutate(season = factor(season)) %>% 
  step_rm(datetime) %>% 
  step_mutate(newtemp = (temp + atemp)/2) %>% 
  step_rm(temp, atemp) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())
preppen_recipe <- prep(penalized_recipe)
bakedpen <- bake(preppen_recipe, new_data = testData)

# Penalized regression model
preg_model <- linear_reg(penalty=0, mixture=1) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
add_recipe(penalized_recipe) %>%
add_model(preg_model) %>%
fit(data=cleandata)
penpreds <- predict(preg_wf, new_data=testData)
penpreds <- exp(penpreds)

## Kaggle Submission 4
kaggle_submission4 <- penpreds %>%
  bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission4, file="./PenPreds.csv", delim=",")
