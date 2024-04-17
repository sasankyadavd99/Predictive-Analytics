#loading library ----
library(psych)
library(corrplot)
install.packages("gtsummary")
library(gtsummary) 
install.packages("caTools")
library(caTools)
library("dplyr")
library(tidyverse)
library(caret)
install.packages("party")
library(party)
library(pROC)
library(tidyverse)
install.packages("randomForest")
library(randomForest)
install.packages("gbm")
library(gbm)
library(e1071)
setwd('C:\\Users\\chait\\Downloads')
#loading hr_data ----
hr_data <-read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv",stringsAsFactors = TRUE)
hr_data
#cleaning the hr_data ----
hr_data[hr_data==""]<- NA
hr_data<-na.omit(hr_data)
hr_data
#summary about the hr_data ----
summary(hr_data)
sapply(hr_data,class)
dim(hr_data)
psych::describe(hr_data)
str(hr_data)
#subset analysis ----
subset_analysis_attrited<-subset(hr_data,Attrition == "Yes")

describe(subset_analysis_attrited)
subset_analysis_females_atttrited <-subset(hr_data,Attrition == "Yes" & 
                                             Gender =="Female")
describe(subset_analysis_females_atttrited)
subset_analysis_males_atttrited <-subset(hr_data,Attrition == "Yes" & 
                                           Gender =="Male" )
describe(subset_analysis_males_atttrited)
subset_traveling<-subset(hr_data,Attrition == "Yes" & 
                           BusinessTravel =="Travel_Frequently" )
describe(subset_traveling)
#histogram for daily rate ----
hist(hr_data$DailyRate,col =rainbow(6),xlab="Daily Rate", ylab="Frequency",main="Total hr_dataset ")
hist(subset_analysis_attrited$DailyRate,col=rainbow(6),xlab="Daily Rate", ylab="Frequency",main="Attrited Employees")
#boxplot for total working years ----
allemployees<-boxplot(hr_data$TotalWorkingYears,main="Boxplot for total working years ")
allemployees
attritedemployees<-boxplot(subset_analysis_attrited$TotalWorkingYears,main="Boxplot for total working years for attrited employees")
attritedemployees
#histogram for percent salary hike
hist(hr_data$PercentSalaryHike,col =rainbow(6),xlab="Salary Hike", ylab="Frequency",main="Salary Hike")
hist(subset_analysis_attrited$PercentSalaryHike,col=rainbow(6),xlab="Salary Hike", ylab="Frequency",main="Salary Hike for attrited employee")
#correlation table ----
num_cols <- hr_data %>% select_if(is.numeric)

# Check for zero standard deviation
zero_sd_vars <- which(sapply(num_cols, sd) == 0)

# Remove variables with zero standard deviation
num_cols_without_zero_sd <- num_cols[-zero_sd_vars]

# Calculate correlation matrix
corr_matrix <- cor(num_cols_without_zero_sd, method = "pearson")
corrplot(corr_matrix)

#mutlivariable and logistic regression method  ----
hr_data$Attrition <- as.factor(hr_data$Attrition)
hr_data$BusinessTravel <- as.factor(hr_data$BusinessTravel)
hr_data$Department <- as.factor(hr_data$Department)
hr_data$EducationField <- as.factor(hr_data$EducationField)
hr_data$Gender <- as.factor(hr_data$Gender)
hr_data$JobRole <- as.factor(hr_data$JobRole)
hr_data$MaritalStatus <- as.factor(hr_data$MaritalStatus)
hr_data$OverTime <- as.factor(hr_data$OverTime)

hr_data <- hr_data %>% 
  select(-c(EmployeeNumber, StandardHours, Over18))
# Split the data into training and test sets
set.seed(123)
trainIndex <- createDataPartition(hr_data$Attrition, p = .8, list = FALSE)
train_data <- hr_data[trainIndex, ]
test_data <- hr_data[-trainIndex, ]
# Fit a logistic regression model using all available variables
logreg_model <- glm(Attrition ~ ., data = train_data, family = "binomial")
par(mfrow=c(2,2))
plot(logreg_model)

# Examine the coefficients to see which variables are most strongly associated with attrition
summary(logreg_model)
test_data$Attrition <- as.factor(test_data$Attrition)
# Use the model to predict attrition on the test set
logreg_predictions <- predict(logreg_model, newdata = test_data, type = "response")
logreg_predictions
test_pred <- ifelse(logreg_predictions > 0.5, "Yes", "No")
test_pred
logreg_confusion_matrix <- table(logreg_predictions, test_data$Attrition)
logreg_accuracy <- sum(diag(logreg_confusion_matrix)) / sum(logreg_confusion_matrix)

logreg_precision <- logreg_confusion_matrix[2, 2] / sum(logreg_confusion_matrix[, 2])
logreg_recall <- logreg_confusion_matrix[2, 2] / sum(logreg_confusion_matrix[2, ])
logreg_f1_score <- 2 * (logreg_precision * logreg_recall) / (logreg_precision + logreg_recall)
# Evaluate the performance of the model using a confusion matrix and ROC curve
confusionMatrix(logreg_predictions, test_data$Attrition)
roc<-roc(test_data$Attrition, logreg_predictions)
roc
par(mfrow=c(1,1))
plot(roc)

#randomforest 
rf_model <- randomForest(Attrition ~ ., data = train_data, ntree = 100)
predictions <- predict(rf_model, newdata = test_data)
confusion_matrix <- table(predictions, test_data$Attrition)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)
confusion_matrix
accuracy
precision
recall
f1_score


#gbm ----
# Convert 'Attrition' variable to binary
train<-train_data
test<-test_data
train$Attrition <- ifelse(train$Attrition == "Yes", 1, 0)
test$Attrition <- ifelse(test$Attrition == "Yes", 1, 0)
# Remove 'EmployeeCount' variable from the dataset
train <- train_data[, !(names(train) == "EmployeeCount")]
test <- test_data[, !(names(test) == "EmployeeCount")]
# Convert 'Attrition' variable to binary
train$Attrition <- as.integer(train$Attrition == "Yes")
test$Attrition <- as.integer(test$Attrition == "Yes")

gbm_model <- gbm(Attrition ~ ., data = train, distribution = "bernoulli", n.trees = 100, interaction.depth = 4, shrinkage = 0.1, cv.folds = 5)
gbm_predictions <- predict(gbm_model, newdata = test, type = "response")
# Convert probabilities to binary predictions
binary_predictions <- ifelse(gbm_predictions > 0.5, 1, 0)

# Create the confusion matrix
confusion_matrixgb <- table(Actual = test$Attrition, Predicted = binary_predictions)

# Calculate accuracy
accuracygb <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Calculate precision
precisiongb <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])

# Calculate recall
recallgb <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
# Calculate F1 score
f1_score_gb <- 2 * (precision * recall) / (precision + recall)


#comparing models 
comparison <- data.frame(
  Model = c("Random Forest", "Logistic Regression","gbm"),
  Accuracy = c(accuracy, logreg_accuracy,accuracygb),
  Precision = c(precision, logreg_precision,precisiongb),
  Recall = c(recall, logreg_recall,recallgb),
  F1_Score = c(f1_score, logreg_f1_score,f1_score_gb )
)
#comaprison
comparison
