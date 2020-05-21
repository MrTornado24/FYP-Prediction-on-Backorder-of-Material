#load my packages
library(devtools)
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(unbalanced)
library(tibble)
library(mlr)
library(ROCR)
library(rpart)
library(randomForest)
library(rpart.plot)
library('ggfortify')
library('devtools')
library('ggplot2')
library('factoextra')
library('FactoMineR')
set.seed(42)
Data <- read.csv('/Users/letv/Desktop/FYP/dataset/Kaggle_Dataset_v2.csv', header = TRUE)
#seeking NA values (Let's see if there are missing values within the dataset)
any(is.na(Data))

#Let's replace the missing values by -99
Data<- na.omit(Data)
#Creating dummy variables : replacing 'No/Yes' to 0/1
Data$potential_issue<-ifelse(Data$potential_issue=='Yes', 1,0)
Data$went_on_backorder<-ifelse(Data$went_on_backorder=='Yes', 1,0)
Data$deck_risk<-ifelse(Data$deck_risk=='Yes', 1,0)
Data$oe_constraint<-ifelse(Data$oe_constraint=='Yes', 1,0)
Data$ppap_risk<-ifelse(Data$ppap_risk=='Yes', 1,0)
Data$stop_auto_buy<-ifelse(Data$stop_auto_buy=='Yes', 1,0)
Data$rev_stop<-ifelse(Data$rev_stop=='Yes', 1,0)

#view cleaned data
str(Data)

#Count : how many parts went on backorders
table(Data['went_on_backorder'])

#Create a Bar Chart
# Create data for the graph.
x <-  c(1676568,11293)
barplot(x,width = 1, main= "Bar Chart", ylab= "Went on Backorder",names.arg= c("No","Yes"),col="blue")
# Create Box plot
boxplot(Data$lead_time,Data$went_on_backorder, ylim= c(0,60), ylab= "Lead Time", main= "Boxplot")
boxplot(Data$forecast_3_month,Data$went_on_backorder, ylab= "forecast_3_month", main= "Boxplot")
boxplot(Data$forecast_6_month,Data$went_on_backorder, ylab= "forecast_6_month", main= "Boxplot")
boxplot(Data$forecast_9_month,Data$went_on_backorder, ylab= "forecast_9_month", main= "Boxplot")
plot.new()
# split the data to training and test
training_split <- Data[c(1:1600000),c(2:24)]
test_split <- Data[c(1600000:1800000), c(2:24)]
View(training_split)

# Use SMOTE sampling to balance the dataset
training_split$went_on_backorder <- as.factor(training_split$went_on_backorder)
input  <- training_split %>% select(-went_on_backorder)
output <- training_split$went_on_backorder 
training_balanced <- ubSMOTE (input,output,  perc.over = 100, perc.under = 100, k = 5)
# Recombine the synthetic balanced data
training_df <- bind_cols(as.tibble(training_balanced$X), tibble(went_on_backorder = training_balanced$Y))

# Plot the two major components using PCA, giving a general interpretation of the dataset
df <- training_df[sample(nrow(training_df), 10000), ]
df$went_on_backorder <- as.factor(df$went_on_backorder)

df.pca <- prcomp(df[, c(2, 3, 17)], center = TRUE, scale = TRUE)
plot.new()
fviz_pca_biplot(df.pca, geom=c("point", "text"), 
                label = "var", 
                
                alpha.var ="contrib", col.var = "contrib",
                fill.ind = df$went_on_backorder, col.ind = "white",
                pointshape = 21, pointsize = 3
)+
  labs(fill = "went_on_order") # Change legend title

# count new dataset after applying SMOTE
View(training_df)
table(training_df['went_on_backorder'])
Y <- c(11293,22586)
percent<- round(100*Y/sum(Y), 1)
percent
# Barplot to vizualize dataset after applying SMOTE
plot.new()
barplot(Y,width = 1, main= "Bar Chart", ylab= "Went on Backorder",names.arg= c("No","Yes"),col="red")

# LOGISTIC REGRESSION MODEL
logistic_regression= glm(went_on_backorder~lead_time+national_inv+forecast_6_month+forecast_3_month+forecast_9_month+
                           sales_1_month+sales_3_month+sales_6_month+sales_9_month+min_bank +
                           local_bo_qty+perf_12_month_avg+potential_issue+deck_risk+oe_constraint
                         +stop_auto_buy, data=training_df, family= binomial)
summary(logistic_regression)
# predict the outcoume on the test data   
predict_test <- predict(logistic_regression, test_split, type= "response")
summary(predict_test)
tapply(predict_test,test_split$went_on_backorder, mean)

#ROC curve to find accurate threshold
ROCR_pred= prediction(predict_test,test_split$went_on_backorder)
ROCRLOG_perf= performance(ROCR_pred, "tpr","fpr")

#Confusion matrix
table(test_split$went_on_backorder,predict_test>0.47)

# AUC
as.numeric(performance(ROCR_pred, "auc")@y.values)

# CART : REGRESSION TREE MODEL

tree_model= rpart(went_on_backorder~lead_time+national_inv+forecast_6_month+forecast_3_month+forecast_9_month+
                    sales_1_month+sales_3_month+sales_6_month+sales_9_month+min_bank +
                    local_bo_qty+perf_12_month_avg+potential_issue+deck_risk+oe_constraint
                  +stop_auto_buy, data= training_df, method = "class", control = rpart.control(minbucket = 25))
prp(tree_model)
# prediction
predictCART = predict(tree_model,newdata=test_split, type= "class" )
table(test_split$went_on_backorder,predictCART )

#ROC curve (CART model)
predict_ROC=predict(tree_model, newdata= test_split)
ROCRCart_pred=prediction(predict_ROC[,2],test_split$went_on_backorder)
ROCRCart_perf= performance(ROCRCart_pred, "tpr","fpr")

#AUC (CART model)
as.numeric(performance(ROCRCart_pred, "auc")@y.values)

# RANDOM FOREST MODEL
training_df$went_on_backorder= as.factor(training_df$went_on_backorder)
test_split$went_on_backorder= as.factor(test_split$went_on_backorder)
randomforest_model= randomForest(went_on_backorder~lead_time+national_inv+forecast_6_month+forecast_3_month+forecast_9_month+
                                   sales_1_month+sales_3_month+sales_6_month+sales_9_month+min_bank +
                                   local_bo_qty+perf_12_month_avg+potential_issue+deck_risk+oe_constraint
                                 +stop_auto_buy, data= training_df, nodesize= 25, ntree= 300)
predict_forest= predict(randomforest_model,newdata = test_split)
table(test_split$went_on_backorder,predict_forest)


#ROC curve (Random Forest model)
predict_roc= predict(randomforest_model,newdata = test_split, type = 'prob')
ROCRRF_pred=prediction(predict_roc[,2],test_split$went_on_backorder)
ROCRRF_perf_RF= performance(ROCRRF_pred, "tpr","fpr")
plot.new()
plot(ROCRRF_perf_RF,col = 'red', text.adj=c(-0.2,1,7), type = c("o"), cex = 0.2)
plot(ROCRCart_perf,col = 'grey',text.adj=c(-0.2,1,7), add = TRUE)
plot(ROCRLOG_perf, col = 'blue',text.adj=c(-0.2,1,7), type = c('o'), cex = 0.1, add = TRUE)


#AUC (Random Forest model)
print("AUC score of Logical Regression is:")
print(as.numeric(performance(ROCR_pred, "auc")@y.values))
print("AUC score of Classification Tree is:")
print(as.numeric(performance(ROCRCart_pred, "auc")@y.values))
print("AUC score of Random Forest is:")
print(as.numeric(performance(ROCRRF_pred, "auc")@y.values))
