###############################################################################
#                 Fraud Modelling
#               A survey of models
###############################################################################

# Loading data ----
fraud_data <- read.csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/payment_fraud.csv")


# Load packages ----
list.of.packages <- c("tidyverse", "caret", "ROCR", "kernlab")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(tidyverse)
library(caret)
library(ROCR)
library(kernlab)

# Structure of data ----

str(fraud_data) # Variable types
summary(fraud_data) # Summarising columns
head(fraud_data) # First 10 rows 
dim(fraud_data) #dimension of the data 39 221 rows and 6 columns

# Visualising ----

ggplot(fraud_data, aes(x= factor(label))) + 
  geom_bar(stat = "count", fill = "steelblue") +
  theme_minimal() # Unbalanced dependent variable

ggsave("Distribution of Labels.png")

ggplot(fraud_data, aes(x= paymentMethod)) + 
  geom_bar(stat = "count", fill = "green") +
  ggtitle("Client payment methods") +
  theme_minimal() # Clients use credit cards the most

ggsave("Client payment methods.png")

ggplot(fraud_data, aes(x = accountAgeDays)) +
  geom_histogram() 

ggsave("Account age days.png")

# Creating train and test sets ----

#Dividing into training and testing dataset
set.seed(263) # random seed
index <- sample(nrow(fraud_data),size = nrow(fraud_data)*0.75) # Getting 75%
fraud_train <- fraud_data[index,]
fraud_test <- fraud_data[-index,]
dim(fraud_train) # 29 415 cases
dim(fraud_test)  # 9806 cases

# Fitting a Logistic regression ----

log_model <- glm(label~., data = fraud_train, family = binomial())
summary(log_model)

#score test data set
fraud_test$score <- predict(log_model,type='response',fraud_test)
log_pred <- ROCR::prediction(fraud_test$score, fraud_test$label)
log_perf <- ROCR::performance(log_pred,"tpr","fpr")

# Logistic Regression Model performance ----

log_AUROC <- round(performance(log_pred, measure = "auc")@y.values[[1]]*100, 2)
log_KS <- round(max(attr(log_perf,'y.values')[[1]]-attr(log_perf,'x.values')[[1]])*100, 2)
log_Gini <- (2*log_AUROC - 100)
cat("AUROC: ", log_AUROC,"\tKS: ", log_KS, "\tGini:", log_Gini, "\n")

# Fitting the support vector machines  ----

svm_model <- ksvm(label~., data = fraud_train, kernel = "vanilladot")
svm_score <- predict(svm_model, fraud_test, type = 'decision')
svm_pred <- prediction(svm_score, fraud_test$label)
svm_perf <- performance(svm_pred, measure = "tpr", x.measure = "fpr")

# SVM Model Performance ----

svm_AUROC <- round(performance(svm_pred, measure = "auc")@y.values[[1]]*100, 2)
svm_KS <- round(max(attr(svm_perf,'y.values')[[1]]-attr(svm_perf,'x.values')[[1]])*100, 2)
svm_Gini <- (2*svm_AUROC - 100)
cat("AUROC: ",svm_AUROC,"\tKS: ", svm_KS, "\tGini:", svm_Gini, "\n")

# Plotting ROC Curve ----

#Compare ROC Performance of Models

plot(log_perf, col='blue', lty=1, main='ROCs: Model Performance Comparision') # logistic regression
plot(svm_perf, col='red',lty=2, add=TRUE); # Vanilla 
legend(0.6,0.5,
       c('Logistic','SVM (Vanilla)'),
       col=c('blue','red'),
       lwd=3);
lines(c(0,1),c(0,1),col = "gray", lty = 4 ) # random line


