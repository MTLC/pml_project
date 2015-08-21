## Coursera Practical Machine Learning course project
library(caret)

## Read training data
raw_train <- read.csv("C:/Users/Charlie/Desktop/Coursera Machine Learning/pml-training.csv",
                stringsAsFactors = FALSE,
                na.strings = "Not Available")

head(raw_train)

## Read testing data
raw_test <- read.csv("C:/Users/Charlie/Desktop/Coursera Machine Learning/pml-testing.csv",
                      stringsAsFactors = FALSE,
                      na.strings = "Not Available")

head(raw_test)

## Candidate predictors determined by visual inspection
## Many variables are sparse in the training data and/or entirely NA in the testing data
## Get varnames of candidate predictors
accel_str <- grep("^accel", colnames(raw_train), value = TRUE)
gyros_str <- grep("^gyros", colnames(raw_train), value = TRUE)
magnet_str <- grep("^magnet", colnames(raw_train), value = TRUE)
pitch_str <- grep("^pitch", colnames(raw_train), value = TRUE)
roll_str <- grep("^roll", colnames(raw_train), value = TRUE)
total_str <- grep("^total", colnames(raw_train), value = TRUE)
yaw_str <- grep("^yaw", colnames(raw_train), value = TRUE)

## Get candidate predictors
predictors_train <- raw_train[, c(accel_str, gyros_str, magnet_str, pitch_str, roll_str, total_str, yaw_str)]

## Create train dataset from candidate predictors and outcome
classe <- raw_train$classe
train <- data.frame(predictors_train, classe)
head(train)

## Create testing dataset from candidate predictors
predictors_test <- raw_test[, c(accel_str, gyros_str, magnet_str, pitch_str, roll_str, total_str, yaw_str)]
test <- data.frame(predictors_test)
head(test)

# Create training and testing sets
inTrain <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]

## Check for near zero covariates
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv

## Work with sample to reduce cycle time
set.seed(9164)
in_sample <- createDataPartition(training$classe, p = 0.1, list = FALSE)
training_sample <- training[in_sample, ]

## Random forests ##########################################################################################
mod_rf <- train(classe ~ ., data=training_sample, method="rf", prox=TRUE)
mod_rf

##Random Forest 

##1376 samples
##52 predictor
##5 classes: 'A', 'B', 'C', 'D', 'E' 

##No pre-processing
##Resampling: Bootstrapped (25 reps) 
##Summary of sample sizes: 1376, 1376, 1376, 1376, 1376, 1376, ... 
##Resampling results across tuning parameters:
    
##    mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
##2    0.8998751  0.8730330  0.01495327   0.01871567
##27    0.9036334  0.8778074  0.01158853   0.01465041
##52    0.8910013  0.8618410  0.01161677   0.01481572

##Accuracy was used to select the optimal model using  the largest value.
##The final value used for the model was mtry = 27. 

## Apply model to testing set
pred_rf <- predict(mod_rf, testing)
cm_rf <- confusionMatrix(pred_rf, testing$classe)
cm_rf

##Confusion Matrix and Statistics

##Reference
##Prediction    A    B    C    D    E
##A 1622   85    1   15    0
##B   14 1000   64    0   24
##C    9   47  944   68   15
##D   27    6   17  878   31
##E    2    1    0    3 1012

##Overall Statistics

##Accuracy : 0.9271          
##95% CI : (0.9202, 0.9336)
##No Information Rate : 0.2845          
##P-Value [Acc > NIR] : < 2.2e-16       

##Kappa : 0.9077          
##Mcnemar's Test P-Value : < 2.2e-16       

##Statistics by Class:

##Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9689   0.8780   0.9201   0.9108   0.9353
##Specificity            0.9760   0.9785   0.9714   0.9835   0.9988
##Pos Pred Value         0.9414   0.9074   0.8717   0.9155   0.9941
##Neg Pred Value         0.9875   0.9709   0.9829   0.9825   0.9856
##Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
##Detection Rate         0.2756   0.1699   0.1604   0.1492   0.1720
##Detection Prevalence   0.2928   0.1873   0.1840   0.1630   0.1730
##Balanced Accuracy      0.9725   0.9282   0.9457   0.9472   0.9670

## Apply model to 20 observations in test set
predict(mod_rf, test)

##[1] B A A A A E D D A A B C B A E E A A B B
##Levels: A B C D E

## Boosting with trees ##########################################################################################
mod_gbm <- train(classe ~ ., data=training_sample, method="gbm", verbose=FALSE)
mod_gbm
##Stochastic Gradient Boosting 

##1376 samples
##52 predictor
##5 classes: 'A', 'B', 'C', 'D', 'E' 

##No pre-processing
##Resampling: Bootstrapped (25 reps) 
##Summary of sample sizes: 1376, 1376, 1376, 1376, 1376, 1376, ... 
##Resampling results across tuning parameters:
    
##    interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD  Kappa SD  
##1                   50      0.7134363  0.6366155  0.01965956   0.02516931
##1                  100      0.7666499  0.7042670  0.01741636   0.02198051
##1                  150      0.7930876  0.7378783  0.01889337   0.02366002
##2                   50      0.7937595  0.7385644  0.01885550   0.02361566
##2                  100      0.8404507  0.7978187  0.01398546   0.01724038
##2                  150      0.8607205  0.8235002  0.01552915   0.01931721
##3                   50      0.8328459  0.7881174  0.01586102   0.01962494
##3                  100      0.8661167  0.8303867  0.01920279   0.02393391
##3                  150      0.8803921  0.8484634  0.01692748   0.02116146

##Tuning parameter 'shrinkage' was held constant at a value of 0.1
##Tuning parameter 'n.minobsinnode' was held constant at a value of 10
##Accuracy was used to select the optimal model using  the largest value.
##The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10. 

## Apply model to testing set
pred_gbm <- predict(mod_gbm, testing)
cm_gbm <- confusionMatrix(pred_gbm, testing$classe)
cm_gbm

##Confusion Matrix and Statistics

##Reference
##Prediction    A    B    C    D    E
##A 1594   63    4    8    9
##B   29  994   60   11   62
##C   14   55  940   52   19
##D   30   14   13  879   38
##E    7   13    9   14  954

##Overall Statistics

##Accuracy : 0.911           
##95% CI : (0.9034, 0.9181)
##No Information Rate : 0.2845          
##P-Value [Acc > NIR] : < 2.2e-16       

##Kappa : 0.8874          
##Mcnemar's Test P-Value : < 2.2e-16       

##Statistics by Class:

##Class: A Class: B Class: C Class: D Class: E
##Sensitivity            0.9522   0.8727   0.9162   0.9118   0.8817
##Specificity            0.9801   0.9659   0.9712   0.9807   0.9910
##Pos Pred Value         0.9499   0.8599   0.8704   0.9025   0.9569
##Neg Pred Value         0.9810   0.9693   0.9821   0.9827   0.9738
##Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
##Detection Rate         0.2709   0.1689   0.1597   0.1494   0.1621
##Detection Prevalence   0.2851   0.1964   0.1835   0.1655   0.1694
##Balanced Accuracy      0.9661   0.9193   0.9437   0.9463   0.9364

## Apply model to 20 observations in test set
predict(mod_gbm, test)

## [1] B A B A A E D B A A B C B A E B A B B B
##Levels: A B C D E
