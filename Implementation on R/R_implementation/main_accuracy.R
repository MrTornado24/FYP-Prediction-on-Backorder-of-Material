# Load the different libraries
library(dplyr)
library(caret)
library(ggplot2)
library(imputeTS)
library(caTools)
library(e1071)
library(nnet)
library(rpart)
library(ggpubr)
library("randomForest")
library(factoextra)
library('class')
library('dlookr')
library('dplyr')
library('corrplot')


# Load the dataset. The blank spaces are replaced by NA.
input_data <- read.csv("/Users/letv/Desktop/FYP/dataset/Kaggle_Dataset_v2.csv", na.strings=c("","NA"))

# Omit the rows that contain NA.
input_data<- na.omit(input_data)
colnames(input_data)

# The patient ID is not a feature, so we remove it. We also remove the stroke feature
input_features = input_data[,3:23]
data_without_id = input_data[,3:24]
head(input_features)

# Converting into factor type
input_features$potential_issue <- as.factor(input_features$potential_issue)
input_features$deck_risk <- as.factor(input_features$deck_risk)
input_features$oe_constraint <- as.factor(input_features$oe_constraint)
input_features$ppap_risk <- as.factor(input_features$ppap_risk)
input_features$stop_auto_buy <- as.factor(input_features$stop_auto_buy)
input_features$rev_stop <- as.factor(input_features$rev_stop)


input_features[] <- data.matrix(input_features)


# Figure 1
input_features_sample <- input_features[sample(nrow(input_features), 10000), ]
data_without_id_sample <- data_without_id[sample(nrow(data_without_id), 10000), ]

res.pca <- prcomp(input_features_sample, scale = TRUE)
fviz_eig(res.pca)


# Figure 2
fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



# Section II (a): the pca values
std_dev <- res.pca[1]
sdev <- std_dev$sdev
eig_values <- sdev^2
pca_values <- eig_values/sum(eig_values)
pca_values <- pca_values*100
pca_values


# Figure 3 (a)
fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)



# Figure 3 (b)
groups <- as.factor(data_without_id_sample$went_on_backorder)
fviz_pca_ind(res.pca,
             geom.ind = "point", # show points only (nbut not "text")
             col.ind = groups, # color by groups
             palette = c("#00AFBB", "#FC4E07"),
             
             ellipse.type = "confidence",
             legend.title = "Groups",
             repel = TRUE,
             addEllipses = TRUE # Concentration ellipses
)





# random downsampling
no_of_exps <-1000
data_without_id$went_on_backorder<-ifelse(data_without_id$went_on_backorder=='Yes', 1,0)
minority_class <- data_without_id[data_without_id$went_on_backorder == 1,]
majority_class <- data_without_id[data_without_id$went_on_backorder == 0,]

# Neural network method

logistic_result = c()

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  minority_sample <- minority_class[sample(nrow(minority_class), 548), ]
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$went_on_backorder, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  logistic_regression= glm(went_on_backorder~lead_time+national_inv+forecast_6_month+forecast_3_month+forecast_9_month+
                             sales_1_month+sales_3_month+sales_6_month+sales_9_month+min_bank +
                             local_bo_qty+perf_12_month_avg+potential_issue+deck_risk+oe_constraint
                           +stop_auto_buy, data=train_set, family= binomial)
  pred_lr <- predict(logistic_regression, newdata=test_set, type= "response")
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 22])
  out_labels<-t(out_labels)
  
  cm_lr = table(out_labels, pred_lr)
  
  #accuracy
  n_lr = sum(cm_lr)
  diag_lr = diag(cm_lr)
  accuracy_lr = sum(diag_lr) / n_lr
  accuracy_lr
  
  
  logistic_result[length(logistic_result)+1] = accuracy_lr
}

x <- logistic_result
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
x[x < (0.5)] <- caps[2]
logistic_result<-x


# Decision tree method

dtree_result = c()

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  
  minority_sample <- minority_class[sample(nrow(minority_class), 548), ]
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$went_on_backorder, SplitRatio = 0.70)
  train_set = subset(balanced_dataset, split == TRUE)
  test_set = subset(balanced_dataset, split == FALSE)
  
  
  train_set$went_on_backorder <- factor(train_set$went_on_backorder)
  classifier_dt = rpart(formula =went_on_backorder~ .,
                        data = train_set)
  print(classifier_dt)
  # Predicting the Test set results
  y_pred_dt= predict(classifier_dt, newdata = test_set, type = 'class')
  
  # Output labels
  out_labels<-as.data.frame(test_set[, 22])
  out_labels<-t(out_labels)
  
  # Making the Confusion Matrix
  cm_dt = table(out_labels, y_pred_dt)
  
  #accuracy
  n_dt = sum(cm_dt)
  diag_dt = diag(cm_dt)
  accuracy_dt = sum(diag_dt) / n_dt
  accuracy_dt
  
  dtree_result[length(dtree_result)+1] = accuracy_dt
}

dtree_result




# Random forest method


rforest_result = c()

for (i in 1:no_of_exps) {
  cat("Current experiment: ", i)
  minority_sample <- minority_class[sample(nrow(minority_class), 548), ]
  majority_sample <- majority_class[sample(nrow(majority_class), 548), ]
  balanced_dataset <- rbind(minority_class, majority_sample)
  
  split = sample.split(balanced_dataset$went_on_backorder, SplitRatio = 0.70)
  trainData = subset(balanced_dataset, split == TRUE)
  testData = subset(balanced_dataset, split == FALSE)
  
  
  
  trainData$went_on_backorder <- as.character(trainData$went_on_backorder)
  trainData$went_on_backorder <- as.factor(trainData$went_on_backorder)
  went_on_backorder_rf = randomForest(went_on_backorder~., data=trainData, ntree=100, proximity=T)
  went_on_backorderPred = predict(went_on_backorder_rf, newdata=testData)
  CM = table(went_on_backorderPred, testData$went_on_backorder)
  accuracy = (sum(diag(CM)))/sum(CM)
  accuracy
  
  rforest_result[length(rforest_result)+1] = accuracy
}

rforest_result





logistic_result
dtree_result
rforest_result


x_name <- "Method"
y_name <- "Accuracy"

a1 <- replicate(no_of_exps, "Logistic Regression")
a2 <- replicate(no_of_exps, "Decision Tree")
a3 <- replicate(no_of_exps, "Random Forest")
a <- c(a1,a2,a3)

all_accuracies <- c(logistic_result,dtree_result,rforest_result)
df <- data.frame(a, all_accuracies)
colnames(df) <- c(x_name, y_name)
print(df)
head(df)


# Figure 4
ggdensity(df, x = "Accuracy",
          add = "mean", rug = TRUE,
          color = "Method", fill = "Method",
          palette = c("#0073C2FF", "#FC4E07", "#07fc9e"))


# saving the df
write.csv(df, file = "/Users/letv/Desktop/FYP/my_results.csv")


# Table 1
cat("Mean accuracy of logistic: ", mean(logistic_result))
cat("Mean accuracy of decision tree: ", mean(dtree_result))
cat("Mean accuracy of random forest: ", mean(rforest_result))