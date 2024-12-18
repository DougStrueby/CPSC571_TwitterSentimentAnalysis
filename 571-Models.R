###Download if you haven't###
install.packages("randomForest")
install.packages("e1071")
install.packages("keras")
install.packages("tensorflow")

library(keras)
library(tensorflow)
install_tensorflow()

###Importing Data###
data = read.csv("tweetProcessedData.csv")
data$y = as.factor(data$y)
set.seed(123)

###Splitting into train/test###
index = sample(nrow(data), round(nrow(data) * 0.8), replace = F)

train = data[index, ]
test = data[-index, ]

###Random Forest###
library(randomForest)

#running random forest
rForest = randomForest(y ~., train, n.trees = 100)
summary(rForest)

#creating prediction model
predRForest = predict(rForest, test, type = "prob")
predRForest = ifelse(predRForest[, "1"] > 0.25, 1, 0)

#confusion matrix
cmRForest = table(predRForest, test$y)

#accuracy calculation
sum(diag(cmRForest))/sum(cmRForest)

# Calculate Precision, Recall, and F-measure
precisionR = cmRForest[2, 2] / sum(cmRForest[, 2])
precisionR

recallR = cmRForest[2, 2] / sum(cmRForest[2, ])
recallR

F_measureR = (2 * precisionR * recallR) / (precisionR + recallR)
F_measureR

#Root Mean Squared Error
#Convert factors to numeric if needed
predRForest_numeric <- as.numeric(as.character(predRForest))
test$y_numeric <- as.numeric(as.character(test$y))

#Calculate RMSE
RMSER = sqrt(mean((predRForest_numeric - test$y_numeric)^2))
RMSER

###Logistic Regression###
#running logistic regression
log = glm(y ~., train, family = "binomial")
summary(log)

#prediction model
predLog = predict(log, test, type = "response")
predLog = ifelse(predLog > 0.25, 1, 0)

#confusion matrix
cmLog = table(predLog, test$y)

#accuracy
sum(diag(cmLog))/sum(cmLog)

# Calculate Precision, Recall, and F-measure
precisionL = cmLog[2, 2] / sum(cmLog[, 2])
precisionL

recallL = cmLog[2, 2] / sum(cmLog[2, ])
recallL

F_measureL = (2 * precisionL * recallL) / (precisionL + recallL)
F_measureL

#Root Mean Squared Error
#Convert factors to numeric if needed
predLog_numeric <- as.numeric(as.character(predLog))
test$y_numeric <- as.numeric(as.character(test$y))

#Calculate RMSE
RMSEL = sqrt(mean((predLog_numeric - test$y_numeric)^2))
RMSEL

###SVM###
library(e1071)

#svm model
m = svm(y ~ ., train, kernel = "polynomial", cost = 25)
summary(m)

#prediction model
predSVM = predict(m, test)

#confusion matrix
cmSVM = table(predSVM, test$y)

#accuracy
sum(diag(cmSVM))/sum(cmSVM)
  
# Calculate Precision, Recall, and F-measure
precisionS = cmSVM[2, 2] / sum(cmSVM[, 2])
precisionS

recallS = cmSVM[2, 2] / sum(cmSVM[2, ])
recallS

F_measureS = (2 * precisionS * recallS) / (precisionS + recallS)
F_measureS

#Root Mean Squared Error
#Convert factors to numeric if needed
predSVM_numeric <- as.numeric(as.character(predSVM))
test$y_numeric <- as.numeric(as.character(test$y))

#Calculate RMSE
RMSES = sqrt(mean((predSVM_numeric - test$y_numeric)^2))
RMSES

###LTSM###
library(keras)
library(tensorflow)

#Reloading data
dataLTSM = read.csv("tweetProcessedData.csv")

trainLTSM = dataLTSM[index, ]
testLTSM = dataLTSM[-index, ]

#predictors
predictorsTrain = trainLTSM[,-58]
predictorsTest = testLTSM[,-58]

#response variable
responseTrain = trainLTSM[,58]
responseTest = testLTSM[,58]


#defining the model
model = keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = ncol(predictorsTrain)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

#compiling the model
model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

#fit the model
history = model %>% fit(
  x = as.matrix(predictorsTrain), 
  y = responseTrain,
  epochs = 25,
  batch_size = 128,
  validation_split = 0.2
)

#plotting training history
plot(history)

#prediction
predLTSM_prob = model %>% predict(as.matrix(predictorsTest))
predLTSM_binary = ifelse(predLTSM_prob > 0.25, 1, 0)

#confusion matrix
cmLTSM = table(predLTSM_binary, responseTest)
cmLTSM

#accuracy
sum(diag(cmLTSM)) / sum(cmLTSM)

# Calculate Precision, Recall, and F-measure
precisionLT = cmLTSM[2, 2] / sum(cmLTSM[, 2])
precisionLT

recallLT = cmLTSM[2, 2] / sum(cmLTSM[2, ])
recallLT

F_measureLT = (2 * precisionLT * recallLT) / (precisionLT + recallLT)
F_measureLT


#Root Mean Squared Error
#Convert factors to numeric if needed
predLTSM_prob_numeric <- as.numeric(as.character(predLTSM_prob))
responseTest_numeric <- as.numeric(as.character(responseTest))

RMSES = sqrt(mean((predLTSM_prob_numeric - responseTest_numeric)^2))
RMSES

