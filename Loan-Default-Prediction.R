#Loan Default Prediction Project

rm(list=ls())
#######load package
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(Metrics)
library(caret)
library(ROSE)
library(randomForest)
library(pROC)
library(xgboost)
library(caret)
library("iml")

########load the data
setwd("/Users/gaibaiting/Desktop/Loan Default Prediction Project/2019Q1")
Acquisitions_Variables <- c("LOAN_ID", "ORIG_CHN", "Seller.Name", "ORIG_RT", "ORIG_AMT", "ORIG_TRM", "ORIG_DTE",
                         "FRST_DTE", "OLTV", "OCLTV", "NUM_BO", "DTI", "CSCORE_B", "FTHB_FLG", "PURPOSE", "PROP_TYP",
                         "NUM_UNIT", "OCC_STAT", "STATE", "ZIP_3", "MI_PCT", "Product.Type", "CSCORE_C", "MI_TYPE", "RELOCATION_FLG")

Performance_Variables <- c("LOAN_ID", "Monthly.Rpt.Prd", "Servicer.Name", "LAST_RT", "LAST_UPB", "Loan.Age", "Months.To.Legal.Mat",
                         "Adj.Month.To.Mat", "Maturity.Date", "MSA", "Delq.Status", "MOD_FLAG", "Zero.Bal.Code", 
                         "ZB_DTE", "LPI_DTE", "FCC_DTE","DISP_DT", "FCC_COST", "PP_COST", "AR_COST", "IE_COST", "TAX_COST", "NS_PROCS",
                         "CE_PROCS", "RMW_PROCS", "O_PROCS", "NON_INT_UPB", "PRIN_FORG_UPB_FHFA", "REPCH_FLAG", 
                         "PRIN_FORG_UPB_OTH", "TRANSFER_FLG")

Acquisitions_ColClasses <- c("character", "factor", "character", "numeric", "integer", "factor", "character",
                             "character", "integer", "integer", "integer", "integer", "integer", "factor",
                             "factor", "factor", "integer", "factor", "factor", "character", "integer",
                             "factor", "integer", "integer","factor" )

# Performance_ColClasses <- c( "character", "factor", "character", "numeric","numeric", 
#                              "integer", "integer", "integer", "factor", "character",
#                              "factor", "factor", "character", "factor", "factor", 
#                              "factor", "factor", "factor", "logical", "logical",
#                              "logical", "logical", "logical", "logical", "logical",
#                              "logical", "numeric", "logical", "factor", "logical", 
#                              "factor")

Acquisition <- read.table("Acquisition_2019Q1.txt", sep="|", 
                          col.names=Acquisitions_Variables, 
                          colClasses = Acquisitions_ColClasses,
                          fill=FALSE, 
                          strip.white=TRUE)

Performance <- read.table("Performance_2019Q1.txt", sep="|", 
                          col.names=Performance_Variables, 
                          # colClasses = Performance_ColClasses,
                          fill=FALSE, 
                          strip.white=TRUE)
sapply(Performance, class)
#The performance data contains 3.8 million rows, and each loan_id contains many performance data. 
#I will just keep the lastest performance record of each loan because that shows the latest loan status.
Performance <- Performance[!rev(duplicated(rev(Performance$LOAN_ID))),]
#combone two datasets
combined_data <- merge(Acquisition, Performance, by=c("LOAN_ID"))
#remove some columns that are irrelevant with the project 
#selected columns in performance: Delq.Status, DISP_DT, Zero.Bal.Code 
combined_data <- combined_data[, c(1:25, 35, 37, 41)]

######## Data cleaning
#create new features 
#ORIG_VAL is the original home value
combined_data$ORIG_VAL <- combined_data$ORIG_AMT/(combined_data$OLTV/100)
#combine the borrower credit scoreas and the co-borrower credit score into the minimum credit score CSCORE_MN
combined_data$CSCORE_MN <- pmin(combined_data$CSCORE_B, combined_data$CSCORE_C)

#zip_3: first 3 digit zipcode of the loan origination location
as.data.frame(sort(table(combined_data$ZIP_3), descreasing=T))
#Download source file, unzip and extract into table
ZipCodeSourceFile = "http://download.geonames.org/export/zip/US.zip"
temp <- tempfile()
download.file(ZipCodeSourceFile , temp)
ZipCodes <- read.table(unz(temp, "US.txt"), sep="\t")
unlink(temp)
names(ZipCodes) <- c("CountryCode", "zip", "PlaceName", 
                    "AdminName1", "AdminCode1", "AdminName2", "AdminCode2", 
                    "AdminName3", "AdminCode3", "latitude", "longitude", "accuracy")
ZipCodes <- ZipCodes[, c(2, 10, 11)]
ZipCodes$ZIP_3 <- substr(ZipCodes$zip,1,3)
ZipCodes <- ZipCodes %>% 
  group_by(ZIP_3) %>% 
  summarise(across(everything(), list(mean)))
ZipCodes <- ZipCodes[, c(1,3,4)]
combined_data <- merge(combined_data, ZipCodes, by=("ZIP_3"))
#drop columns
combined_data <- subset(combined_data, select=-c(LOAN_ID, ZIP_3,MI_TYPE,MI_PCT,CSCORE_B,
                                                 CSCORE_C,Product.Type,ORIG_DTE,FRST_DTE)) 

#zero balance code
# 01 = Prepaid or Matured
# 02 = Third Party Sale
# 03 = Short Sale
# 06 = Repurchased
# 09 = Deed-in-Lieu,REO: A deed in lieu of foreclosure is a deed instrument in which a mortgagor (i.e. the borrower) conveys all interest in a real property to the mortgagee (i.e. the lender) to satisfy a loan that is in default and avoid foreclosure proceedings.
# 15 = Note Sale
# 16 = Reperforming Loan Sale
is.na(combined_data$Zero.Bal.Code) <- combined_data$Zero.Bal.Code == ""
table(combined_data$Zero.Bal.Code, useNA='always')
#the 'good' codes 01, 06, 16 look like they have a current healthy loan status, 
#but the data does not tell their final status, a good loan might lead to default if the borrower lost the income and can't pay the debt. 
#I also notice 320944 loans without a ZB code, which means a delinquency occurs
#I will use the delinquency status to determine whether the loan is default if the zero balance code is NA

#delinquency
#0 = Current, or less than 30 days past due, 1=30–59days, 2=60–89, 3=90-119, x=liquite or unknown
table(combined_data$Delq.Status, useNA='always')
#check 'x' : 01-61020 loans, 06-190 loans, 09-11 loans(default)
table(combined_data[combined_data$Delq.Status=='X', ]$Zero.Bal.Code)
#check the realtionship between deliquency status and zeio balance code
relationship <- combined_data %>%
  group_by(Delq.Status, Zero.Bal.Code) %>%
  summarise(size=n())
relationship

######determine 'default'
#for loan with a 'X' deliquency status, if the zero balance code in ('01', '06'), the loan is healthy loan, if the zero balance code is '09', it is a default loan
#for other loans (with NA in zero balance code), I define loan that has more than 1 months deliquency may lead to a default
combined_data$default <- 0
#combined_data[combined_data$Delq.Status=='X' & combined_data$Zero.Bal.Code=='09', "default"] <- 1
combined_data[!combined_data$Delq.Status %in% c('0', 'X'), "default"] <- 1

### very imblanced data
table(combined_data$default)
combined_data$default <- as.factor(combined_data$default)
ggplot(combined_data) + geom_bar(aes(x=default))


#####Explotary data analysis
##categorical data: grouped barplot
colSums(is.na(combined_data))
category_var <- c("ORIG_CHN","Seller.Name","NUM_BO","FTHB_FLG","PURPOSE","NUM_UNIT","PROP_TYP","OCC_STAT","RELOCATION_FLG")

tbl <- with(combined_data, table(default, ORIG_CHN))
ggplot(as.data.frame(tbl), aes(factor(default), Freq, fill=ORIG_CHN)) +
  geom_bar(position="dodge", stat="identity") +
  ggtitle('Default Distribution by Origination Channel') +
  xlab('Default')

num_bo <- with(combined_data, table(default, NUM_BO))
ggplot(as.data.frame(num_bo), aes(factor(default), Freq, fill=NUM_BO)) +
  geom_bar(position="dodge", stat="identity") +
  ggtitle('Default Distribution by Number of Borrower') +
  xlab('Default')

purpose <- with(combined_data, table(default, PURPOSE))
ggplot(as.data.frame(purpose), aes(factor(default), Freq, fill=PURPOSE)) +
  geom_bar(position="dodge", stat="identity") +
  ggtitle('Default Distribution by Purpose') +
  xlab('Default')

occ <- with(combined_data, table(default, OCC_STAT))
ggplot(as.data.frame(occ), aes(factor(default), Freq, fill=OCC_STAT)) +
  geom_bar(position="dodge", stat="identity") +
  ggtitle('Default Distribution by Occupancy Type') +
  xlab('Default')

##default and state
df <- combined_data %>%
  group_by(STATE, default) %>%
  summarise(size=n())

df2 <- df %>%
  group_by(STATE) %>%
  summarise(percent = 100*sum(size[default==1]/sum(size)))

df2 <- head(df2 %>% arrange(desc(percent)), 10)

ggplot(df2, aes(x=reorder(STATE, -percent), y=percent, fill=STATE)) +
  geom_bar(stat="identity") +
  xlab("State") +
  ylab("Default Rate") +
  ggtitle(" Default Rate by State")

##numeric varaibales distribution with differnet levels of default
numeric_var <- c("ORIG_RT","ORIG_AMT","ORIG_TRM","OLTV","OCLTV","DTI","ORIG_VAL","CSCORE_MN")
combined_data$ORIG_TRM <- as.numeric(combined_data$ORIG_TRM)
ggplot(combined_data, aes(x = ORIG_RT)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Original Interest Rate') +
  xlab('Original Interest Rate')
ggplot(combined_data, aes(x = ORIG_AMT)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Original Amount') +
  xlab('Original Amount')
ggplot(combined_data, aes(x = ORIG_TRM)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Original Loan Term') +
  xlab('Original Loan Term')
ggplot(combined_data, aes(x = OLTV)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Original Loan-to- Value (LTV)') +
  xlab('Original Loan-to- Value (LTV)')
ggplot(combined_data, aes(x = DTI)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Debt-to-Income Ratio') +
  xlab('Debt-to-Income Ratio')
ggplot(combined_data, aes(x = ORIG_VAL)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Original Value') +
  xlab('Original Value')
ggplot(combined_data, aes(x = CSCORE_MN)) + geom_density(aes(fill = default), alpha = 0.4) +
  ggtitle('Default Distribution by Minimum Credit Score') +
  xlab('Minimum Credit Score')


#log transformation
combined_data$log_ORIG_VAL <- log(combined_data$ORIG_VAL)
combined_data$log_ORIG_AMT <- log(combined_data$ORIG_AMT)
combined_data$log_OLTV <- log(combined_data$OLTV)
combined_data$log_OCLTV <- log(combined_data$OCLTV)
combined_data$log_DTI <- log(combined_data$DTI)


##default rate and location
df3 <- combined_data %>%
  group_by(latitude_1, longitude_1, default) %>%
  summarise(size=n())

df4 <- df3 %>%
  group_by(latitude_1, longitude_1) %>%
  summarise(percent = 100*sum(size[default==1]/sum(size)))

mid <- median(df4$percent)
ggplot(df4[df4$longitude_1<0, ], aes(longitude_1, latitude_1, color=percent, size = percent*2)) + 
  geom_point() +
  scale_color_gradient2(guide='colourbar')


#####Feature engineering before modeling
##create dummy varaiables
dummy_var <- c("ORIG_CHN","NUM_BO","FTHB_FLG","PURPOSE",
           "NUM_UNIT","PROP_TYP","STATE",'RELOCATION_FLG','OCC_STAT')
combined_data <- fastDummies::dummy_cols(combined_data, select_columns = dummy_var)

##drop columns
drop_columns <- names(combined_data) %in% c('ORIG_AMT', 'ORIG_TRM', 'OLTV', 'OCLTV', 'DTI','ORIG_VAL',
                                            'Zero.Bal.Code','DISP_DT','Delq.Status',
                                            "ORIG_CHN","Seller.Name","NUM_BO","FTHB_FLG","PURPOSE",
                                            "NUM_UNIT","PROP_TYP","STATE",'RELOCATION_FLG','OCC_STAT')
data <- combined_data[!drop_columns]

##########Data Modeling
###split data
set.seed(123)
smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]
table(test$default)

## unbalanaced data
prop.table(table(train$default))

###resampling and decision tree
## the minority class is oversampled with replacement and majority class is undersampled without replacement.
data_balanced_both <- ovun.sample(default ~ ., data = train, method = "both", p=0.5, N=1000, seed = 1)$data
table(data_balanced_both$default)
tree.both <- rpart(default ~ ., data = data_balanced_both, method='class')
pred.tree.both <- predict(tree.both, newdata = test[,!colnames(test) %in% c("default")], type='class')
confusionMatrix(pred.tree.both, test$default)
roc.curve(test$default, pred.tree.both, main='Mixed resampling method')

#undersampling
table(train$default)
data_balanced_under <- ovun.sample(default ~ ., data = train, method = "under", N = 2812, seed = 1)$data
table(data_balanced_under$default)
tree.under <- rpart(default ~ ., data = data_balanced_under, method='class')
pred.tree.under <- predict(tree.under, newdata = test[,!colnames(test) %in% c("default")], type='class')
confusionMatrix(pred.tree.under, test$default)
roc.curve(test$default, pred.tree.under, main='Undersampling method')

#oversampling
data_balanced_over <- ovun.sample(default ~ ., data = train, method = "over")$data
table(data_balanced_over$default)
tree.over <- rpart(default ~ ., data = data_balanced_over, method='class')
pred.tree.over <- predict(tree.over, newdata = test[,!colnames(test) %in% c("default")], type='class')
confusionMatrix(pred.tree.over, test$default)
roc.curve(test$default, pred.tree.over, main='Oversampling method')

rpart.plot(tree.under)


###random forest
#original random forest
rf_old <- randomForest(default~. , data = data_balanced_under, ntree=20)
rf_pred_old <- predict(rf_old, test[,!colnames(test) %in% c("default")])
confusionMatrix(rf_pred_old, test$default)
varImpPlot(rf_old)

#parameter tuning
#grid search 
# tunegrid <- expand.grid(.mtry = (1:15)) 
# 
# rf_gridsearch <- train(default ~ ., 
#                        data = data_balanced_both,
#                        method = 'rf',
#                        metric = 'Accuracy',
#                        tuneGrid = tunegrid)
# print(rf_gridsearch)

set.seed(1)
bestMtry <- tuneRF(data_balanced_under[which(names(data_balanced_under) != "default")], data_balanced_under$default, stepFactor = 1.5, improve = 1e-5, ntree = 20)
print(bestMtry)
plot(bestMtry, type='l')

rf <- randomForest(default~. , data = data_balanced_under, ntree=20, mtry=4)
rf
#prediction
rf_pred <- predict(rf, test[,!colnames(test) %in% c("default")])
rf_pred
#confusion matrix
confusionMatrix(rf_pred, test$default)

# feature importance
imp <- cbind.data.frame(Feature=rownames(rf$importance),rf$importance)
imp <- head(imp %>% arrange(desc(MeanDecreaseGini)), 15)
imp
  
ggplot(imp, aes(x=reorder(Feature, -MeanDecreaseGini), y=MeanDecreaseGini, fill=Feature)) + 
  geom_bar(stat='identity') + 
  xlab('Feature') +
  ggtitle('Featuer importanace')

varImpPlot(rf)

#featur effect
features <- data_balanced_both[which(names(data_balanced_both) != "default")]
predictor.rf <- Predictor$new(
  model = rf, 
  data = features, 
  y = data_balanced_both$default)

cscore <- FeatureEffect$new(predictor.rf, feature = "CSCORE_MN")
cscore$plot()

orig <- FeatureEffect$new(predictor.rf, feature = "ORIG_RT")
orig$plot()

dti <- FeatureEffect$new(predictor.rf, feature = "log_DTI")
dti$plot()

ocltv <- FeatureEffect$new(predictor.rf, feature = "log_OCLTV")
ocltv$plot()

####XgBoost
xgb <- xgboost(data = data.matrix(data_balanced_under[,!colnames(data_balanced_under) %in% c("default")]), 
               label = as.numeric(as.character(data_balanced_under$default)), 
               max.depth = 10, 
               eta = 0.1, 
               nthread = 2, 
               nrounds = 2,
               objective = "binary:logistic")
xgb_pred <- predict(xgb, data.matrix(test[,!colnames(test) %in% c("default")]))
pred <- as.numeric(xgb_pred > 0.5)
err <- mean(as.numeric(pred > 0.5) != test$default)
confusionMatrix(as.factor(pred), test$default)

####model comprasion
#ROC
dt_prediction <- predict(tree.under, test, type='prob')
rf_prediction <- predict(rf, test, type = "prob")
ROC_dt <- roc(test$default, dt_prediction[,2], levels = c(0,1), direction = "<")
ROC_rf <- roc(test$default, rf_prediction[,2], levels = c(0, 1), direction = "<")
ROC_xgb <- roc(test$default, xgb_pred, levels = c(0, 1), direction = "<")
plot(ROC_dt, col = "green", xlim = c(1, 0), main = "ROC Comparison between three models", asp = NA) 
lines(ROC_rf, col = "red")
lines(ROC_xgb, col = 'blue')
legend(0.35, 0.3, legend=c("Decision tree", "Random Forest", "XGBoost"),
       col=c("green", "red", "blue"), lty=1:2, cex=0.8)






