##################################  
# Install packages, download files  
##################################

if (!require(tidyverse)) install.packages("tidyverse")
  library(tidyverse)
if (!require(caret)) install.packages("caret")
  library(caret)
if (!require(purrr)) install.packages("purrr")
  library(purrr)
if (!require(caretEnsemble)) install.packages("caretEnsemble")
  library(caretEnsemble)

#get files from GitHub url
url_adm <- "https://raw.githubusercontent.com/rravikumar11/
gender_ratio_prediction/main/data/adm2018.csv"
url_hd <- "https://raw.githubusercontent.com/rravikumar11/
gender_ratio_prediction/main/data/hd2018.csv"
url_sfa <- "https://raw.githubusercontent.com/rravikumar11/
gender_ratio_prediction/main/data/sfa1718.csv"
adm <- read.csv(url_adm)
hd <- read.csv(url_hd)
sfa <- read.csv(url_sfa)



################### 
# Cleaning the data  
###################

#merging the datasets, generating response variable y
edu <- adm %>% select(UNITID, APPLCN, ADMSSN, ENRLT) %>% 
  inner_join(hd, by = "UNITID") %>% inner_join(sfa, by = "UNITID") 
edu <- edu %>% filter(!is.na(APPLCN)) %>% filter(!is.na(ADMSSN))
adm_prob <- edu$ADMSSN/edu$APPLCN
edu$y <- adm_prob
edu <- edu %>% select(-UNITID, -APPLCN, -ADMSSN)

#converting ZIP code to numeric
edu <- edu %>% mutate(ZIP = as.numeric(substr(ZIP, 1, 5)))

#removing variables unique to each institution
edu <- edu %>% select(-INSTNM, -IALIAS, -ADDR, -FIPS, -CHFNM, -GENTELE, 
                      -EIN, -DUNS, -OPEID, -WEBADDR, -ADMINURL, -FAIDURL, -APPLURL, 
                      -NPRICURL, -VETURL, -ATHURL, -DISAURL, -CITY, -STABBR, -CHFTITLE, 
                      -F1SYSNAM, -COUNTYNM)

#removing variables containing "X"
edu <- edu %>% select(-contains("X"))

#remove all variables with >1300 NA values
nacount <- map(edu, ~sum(is.na(.))) 
fewna <- which(nacount > 1300) 

#remove all observations with NAs
edu <- na.omit(edu[,-fewna]) 

#remove all remaining non-varying variables
uniques <- sapply(edu, unique) 
uniquelen <- sapply(uniques, length) 
whichone <- which(uniquelen == 1) 
edu <- edu[,-whichone] 

#graphing admissions rate by school size (check if regularization is needed)
edu %>% ggplot(aes(ENRLT, y)) + geom_point()

#creating the validation subset
set.seed(1, sample.kind = "Rounding")
validation_index <- createDataPartition(edu$y, times = 1, p = 0.2, list = FALSE)
edu_validation <- edu[validation_index,]
edu_model <- edu[-validation_index,]

#creating the train and test subsets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edu_model$y, times = 1, p = 0.4, list = FALSE)
edu_test <- edu[test_index,]
edu_train <- edu[-test_index,]



#########################################  
# OLS, random forest, important variables  
#########################################

#define function to calculate RMSE
calc_RMSE <- function(y_hat, y) {
  sqrt(mean((y_hat - y)^2))
}

#OLS model (all predictors)
train_ols <- lm(y ~ ., data = edu_train)
y_hat_ols <- predict(train_ols, edu_test)
RMSE_ols <- calc_RMSE(y_hat_ols, edu_test$y)
rmse_table <- data.frame(model = "OLS", rmse = RMSE_ols)

#RF model (all predictors) NOTE: This segment of code can take a long time to run
set.seed(1, sample.kind = "Rounding")
train_rf <- train(y ~ ., method = "rf", data = edu_train)
y_hat_rf <- predict(train_rf, edu_test)
RMSE_rf <- calc_RMSE(y_hat_rf, edu_test$y)

#updating table of RMSEs, including tune column
rmse_table <- rbind(rmse_table, data.frame(model = "RF", rmse = RMSE_rf))
rmse_table <- cbind(rmse_table, data.frame(
  tune = c("--", paste("mtry = ", train_rf$bestTune))))

#vector of variables by importance
imp <- varImp(train_rf)$importance
imp <- imp %>% arrange(desc(Overall))
head(imp, 20)

#selecting 20 most important predictors
edu_validation_select <- edu_validation %>% 
  select(GRN4A22,ANYAIDP,AIDFSIP,AGRNT_P,LATITUDE,GRN4A32,GRN4A21,
         NPT452,GRN4A12,GRN4A20,IGRNT_P,ZIP,FLOAN-P,NPT452,GRN4A31,
         C18UGPRF,OLOAN_A,LONGITUD,NPT452,LOAN_P,y)
edu_test_select <- edu_test %>% 
  select(GRN4A22,ANYAIDP,AIDFSIP,AGRNT_P,LATITUDE,GRN4A32,GRN4A21,
         NPT452,GRN4A12,GRN4A20,IGRNT_P,ZIP,FLOAN-P,NPT452,GRN4A31,
         C18UGPRF,OLOAN_A,LONGITUD,NPT452,LOAN_P,y)
edu_train_select <- edu_train %>% 
  select(GRN4A22,ANYAIDP,AIDFSIP,AGRNT_P,LATITUDE,GRN4A32,GRN4A21,
         NPT452,GRN4A12,GRN4A20,IGRNT_P,ZIP,FLOAN-P,NPT452,GRN4A31,
         C18UGPRF,OLOAN_A,LONGITUD,NPT452,LOAN_P,y)



#######################################  
# Reduced OLS/RF/k-NN, Cubist, Ensemble  
#######################################

#OLS model (select predictors)
train_ols_select <- lm(y ~ ., data = edu_train_select)
y_hat_ols_select <- predict(train_ols_select, edu_test_select)
RMSE_ols_select <- calc_RMSE(y_hat_ols_select, edu_test_select$y)
rmse_table <- rbind(rmse_table, data.frame(
  model = "OLS (select)", rmse = RMSE_ols_select, tune = "--"))


#RF model (select predictors)
set.seed(17, sample.kind = "Rounding")
train_rf_select <- train(y ~ ., method = "rf", data = edu_train_select, 
                         tuneGrid = data.frame(mtry = c(1,10)))
y_hat_rf_select <- predict(train_rf_select, edu_test_select)
RMSE_rf_select <- calc_RMSE(y_hat_rf_select, edu_test_select$y)
rmse_table <- rbind(rmse_table, data.frame(
  model = "RF (select)", rmse = RMSE_ols_select, tune = paste("mtry = ", train_rf_select$bestTune)))


#k-NN model (select predictors)
set.seed(28, sample.kind = "Rounding")
train_knn_select <- train(y ~ ., method = "knn", data = edu_train_select, 
                          tuneGrid = data.frame(k = seq(20, 30, 1)))
y_hat_knn_select <- predict(train_knn_select, edu_test_select)
RMSE_knn_select <- calc_RMSE(y_hat_knn_select, edu_test_select$y)
rmse_table <- rbind(rmse_table, data.frame(
  model = "k-NN (select)", rmse = RMSE_knn_select, tune = paste("k = ", train_knn_select$bestTune)))


#Cubist model (all predictors)
set.seed(5, sample.kind = "Rounding")
train_cubist <- train(y ~ ., method = "cubist", data = edu_train, 
                      tuneGrid = data.frame(
                        committees = seq(16,24,2), neighbors = c(0:4)))
y_hat_cubist <- predict(train_cubist, edu_test)
RMSE_cubist <- calc_RMSE(y_hat_cubist, edu_test$y)
rmse_table <- rbind(rmse_table, data.frame(model = "Cubist", rmse = RMSE_cubist, tune = paste(
    "committees = ", train_cubist$bestTune$committees, " / neighbors = ", 
    train_cubist$bestTune$neighbors)))


#Cubist model (select predictors)
set.seed(5, sample.kind = "Rounding")
train_cubist_select <- train(y ~ ., method = "cubist", data = edu_train_select, 
                             tuneGrid = data.frame(
                               committees = seq(16,24,2), neighbors = c(0:4)))
y_hat_cubist_select <- predict(train_cubist_select, edu_test_select)
RMSE_cubist_select <- calc_RMSE(y_hat_cubist_select, edu_test_select$y)
rmse_table <- rbind(rmse_table, data.frame(model = "Cubist", rmse = RMSE_cubist_select, tune = paste(
  "committees = ", train_cubist_select$bestTune$committees, " / neighbors = ", 
  train_cubist_select$bestTune$neighbors)))


#Ensemble model (select predictors)
ensemble_select_model_list <- caretList(y ~ ., data = edu_train_select, tuneList = list(
  rf = caretModelSpec(method = "rf", tuneGrid = data.frame(mtry = 1)),
  ols = caretModelSpec(method = "lm"),
  knn = caretModelSpec(method = "knn", tuneGrid = data.frame(k = 22)),
  cubist = caretModelSpec(method = "cubist", tuneGrid = data.frame(
    committees = 16, neighbors = 0))))
y_hat_ensemble_select <- predict(ensemble_select_model_list, edu_test_select)
RMSE_ensemble_select <- calc_RMSE(y_hat_ensemble_select, edu_test_select$y)
rmse_table <- rbind(rmse_table, data.frame(model = "Ensemble (select)", rmse = RMSE_ensemble_select, tune = "--"))

#Ensemble model (full predictors)
ensemble_model_list <- caretList(y ~ ., data = edu_train, tuneList = list(
  rf = caretModelSpec(method = "rf", tuneGrid = data.frame(mtry = 110)),
  cubist = caretModelSpec(method = "cubist", tuneGrid = data.frame(
    committees = 16, neighbors = 0))))
y_hat_ensemble <- predict(ensemble_model_list, edu_test)
RMSE_ensemble <- calc_RMSE(y_hat_ensemble, edu_test$y)
rmse_table <- rbind(rmse_table, data.frame(model = "Ensemble", rmse = RMSE_ensemble_select, tune = "--"))
