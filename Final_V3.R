####################################################
#Sami Chowdhury
#423006267
#8/2/19
####################################################

####################################################
##load libraries, entropy file, and seed
####################################################

library(tidytext)
library(tidyr)
library(dplyr)
library(stringr)
library(caret)
library(ggplot2)
library(sentimentr)
library(tm)
library(gmodels)
library(caTools)
library(e1071)
library(ggplot2)
library(caret)
library(tree)
library(party)
library(class)
library(MASS)
library(nnet)

set.seed(145)

source('shannon.entropy.R')
source('function.appendix.R')

data(stop_words)

####################################################
##read data into dfs (twice because of diff manips)
####################################################
mydf_v3_1 = readr::read_csv('hate.speech.csv') #for sentiment analysis
mydf_v3_2 = readr::read_csv('hate.speech.csv') #for entropy and tf




########################################################################################################
####################################################
# PART ONE (A): SENTIMENT ANALYSIS######################################################################
####################################################
########################################################################################################




####################################################
##initialize table with standard vals for sentiment detection
####################################################

mydf_v3_1$sentiment_score = apply(mydf_v3_1,1,function(x){get_sentence_score(x[3])})
mydf_v3_1$sentiment_decision = "regular"

####################################################
##add sentiments to the table
####################################################

for(i in 1:nrow(mydf_v3_1)){
  if(mydf_v3_1$sentiment_score[i] >0)
  {
    mydf_v3_1$sentiment_decision[i] = "positive"
  }
  if(mydf_v3_1$sentiment_score[i] <0)
  {
    mydf_v3_1$sentiment_decision[i] = "negative"
  }
}



########################################################################################################
####################################################
## PART ONE (B): ENTROPY SECTION########################################################################
####################################################
########################################################################################################



####################################################
## get rid of stop words
####################################################

clean_tweets = mydf_v3_2 %>% unnest_tokens(word, tweet_text) %>% filter(!str_detect(word, "[0-9]")) %>% anti_join(stop_words) %>% ungroup()

####################################################
##get frequencies as per entropy.lab.r 
####################################################

freq = clean_tweets %>% group_by(tweet_id, speech) %>% mutate(word = str_extract(word, "[a-z']+")) %>% count(word, sort = T) %>% ungroup() %>% mutate(probability = n / sum(n)) %>% select(-n) 

freq[is.na(freq)] = 0

####################################################
##apply shannon.entropy function to the frequencies/probabilities, with my functions
####################################################

mydf_v3_1$entropy_score = apply(mydf_v3_1,1,function(x,y){get_entropy_sentence(x[3], freq)})



########################################################################################################
####################################################
## PART TWO: TF_IDF_SECTION#############################################################################
####################################################
########################################################################################################




####################################################
##tidy the tweets for tf calculation as per tidy3 lab
####################################################

tweet_words = mydf_v3_2 %>% unnest_tokens(word, tweet_text) %>% count(tweet_id, word, sort=T) %>% ungroup()

total_words = tweet_words %>% group_by(tweet_id) %>% summarize(total = sum(n))

tweet_words = left_join(tweet_words, total_words)

tf_words = tweet_words %>% mutate(tf = n / total)

####################################################
##use bind_tf_idf to get idf
####################################################

tfidf_words = tweet_words %>% bind_tf_idf(word, tweet_id, n)

####################################################
##use my tf and put into table, then calc tfidf
####################################################

tfidf_words$tf = tf_words$tf
tfidf_words = tfidf_words %>% mutate(tfidf_mine = tf * idf)

tfidf_words = tfidf_words %>% group_by(tweet_id) %>% mutate(total_tf_idf = sum(tfidf_mine)) %>% ungroup()

tfidf_words = tfidf_words %>% group_by(tweet_id) %>% slice(1) %>% ungroup()



########################################################################################################
####################################################
## PART THREE (A): SVM RADIAL BUILDING##################################################################
####################################################
########################################################################################################



####################################################
##build the 2 feature sets
####################################################

set1 = data.frame(mydf_v3_1$tweet_id, mydf_v3_1$tweet_text, mydf_v3_1$sentiment_decision, mydf_v3_1$entropy_score)

####################################################
#order the mydf so that tweet_ids and tweet_text match to the tfidf calculated
####################################################

mydf_v3_2 = mydf_v3_2[order(mydf_v3_2$tweet_id),]

set2 = data.frame(mydf_v3_2$tweet_id, mydf_v3_2$tweet_text, tfidf_words$total_tf_idf)

####################################################
##create the output set, then make the dataframes
####################################################

y = mydf_v3_2$speech

data_me1 = data.frame(x=set1, y=as.factor(y))
data_me2 = data.frame(x=set2, y=as.factor(y))

####################################################
#split the data into test and train
####################################################

sample1 = sample.int(n = nrow(data_me1), size = floor(0.8*nrow(data_me1)), replace=F)
sample2 = sample.int(n = nrow(data_me2), size = floor(0.8*nrow(data_me2)), replace=F)

train1 = data_me1[sample1,]
test1 = data_me1[-sample1,]

train2 = data_me2[sample2,]
test2 = data_me2[-sample2,]

####################################################
#fit with svm function - first set of feautres
####################################################

y = data.frame(y)

mymodel_set1 = svm(y~., data = train1, kernel = "radial", gamma = 1, cost = 1)

####################################################
##Find the best gamma and cost: FIRST SET
####################################################

tune.out = tune(svm, y~., data = train1, kernel = "radial", ranges = list(cost = 10^(-1:2), gamma = c(0.5, 1:4)))

####################################################
##Summary state and tuning FIRST SET
####################################################

summary(tune.out)

#after my first run of the tune.out, best performance: 0.6649233, cost 10, gamma 3

ypred = predict(tune.out$best.model, newdata = test1)

result1 = table(predict = ypred, truth = test1$y)

####################################################
##SVM fitting, second set of features
####################################################

mymodel_set2 = svm(y~., data = train2, kernel = "radial", gamma = 1, cost = 1)

####################################################
##Find the best gamma and cost: SECOND SET
####################################################

tune.out2 = tune(svm, y~., data = train2, kernel = "radial", ranges = list(cost = 10^(-1:2), gamma = c(0.5, 1:4)))

####################################################
##Summary state and tuning SECOND SET
####################################################

summary(tune.out2)

#best cost 0.1, gamma 0.5, perf 0.5835252

ypred2 = predict(tune.out2$best.model, newdata = test2)

result2 = table(predict = ypred, truth = test2$y)


########################################################################################################
####################################################
##PART THREE (B): KNN BUILDING #########################################################################
####################################################
########################################################################################################


####################################################
#make the knn datasets
####################################################

knn_set1 = data.frame(data_me1, stringsAsFactors = FALSE)
knn_set2 = data.frame(data_me2, stringsAsFactors = FALSE)

colnames(knn_set1) = c("ID","text","sentiment","score","y")
colnames(knn_set2) = c("ID","text", "tfidf","y")

####################################################
#factorize the dependent variable, or speech sentiment
####################################################

knn_set1$y = factor(knn_set1$y)
knn_set2$y = factor(knn_set2$y)
knn_set2$tfidf[is.na(knn_set2$tfidf)] = 0

####################################################
#normalize numeric variables
####################################################

numerics_var = sapply(knn_set1, is.numeric)

knn_set1[numerics_var] = lapply(knn_set1[numerics_var], scale)

#knn can only allow booleans or numericas, convert sentiment decision to numericas

knn_set1$senty = 0.5
for(i in 1:nrow(knn_set1)){
  if(as.character(knn_set1$sentiment[i])=="positive")
  {
    knn_set1$senty[i] = 1
  }
  if(as.character(knn_set1$sentiment[i])=="negative")
  {
    knn_set1$senty[i] = 0
  }
}

#scale tfidfs serpeately since otherwise i get attributes returned
knx = scale(knn_set2$tfidf, center=TRUE,scale=TRUE)

knn_set2$tfidf_scaled = knx

####################################################
#get only independent variables
####################################################

ind_vars1 = c("senty", "score")
knn_sub1 = knn_set1[ind_vars1]

knn_sub2 = knn_set2$tfidf_scaled

####################################################
#predict on set of about 1000 observations
####################################################

set.seed(223)
ind = 1:100

####################################################
#KNN First Feature Set
####################################################

knn_train1 = knn_sub1[-ind,]
knn_test1 = knn_sub1[ind,]

knn_sentiment_train1 = knn_set1$y[-ind]
knn_sentiment_test1 = knn_set1$y[ind]

knn.1set1 = knn(knn_train1, knn_test1, knn_sentiment_train1, k=1)
knn.5set1 = knn(knn_train1, knn_test1, knn_sentiment_train1, k=5)
knn.20set1 = knn(knn_train1, knn_test1, knn_sentiment_train1, k=20)

####################################################
#calc correct classification
####################################################

correc1_set1 = 100 * sum(knn_sentiment_test1 == knn.1set1) / 100
correc5_set1 = 100 * sum(knn_sentiment_test1 == knn.5set1) / 100
correc20_set1 = 100 * sum(knn_sentiment_test1 == knn.20set1) / 100

####################################################
#KNN Second Feature Set
####################################################

knn_train2 = knn_sub2[-ind,]
knn_test2 = knn_sub2[ind,]

knn_sentiment_train2 = knn_set2$y[-ind]
knn_sentiment_test2 = knn_set2$y[ind]

knn.1set2 = knn(data.frame(knn_train2), data.frame(knn_test2), knn_sentiment_train2, k=1)
knn.5set2 = knn(data.frame(knn_train2), data.frame(knn_test2), knn_sentiment_train2, k=5)
knn.20set2 = knn(data.frame(knn_train2), data.frame(knn_test2), knn_sentiment_train2, k=20)

####################################################
#calc correct classification
####################################################

correc1_set2 = 100 * sum(knn_sentiment_test2 == knn.1set2) / 100
correc5_set2 = 100 * sum(knn_sentiment_test2 == knn.5set2) / 100
correc20_set2 = 100 * sum(knn_sentiment_test2 == knn.20set2) / 100

####################################################
#Pooling all knn results together
####################################################


#how the classifications went

knn_result_1set1 = table(knn.1set1, knn_sentiment_test1)
knn_result_5set1 = table(knn.5set1, knn_sentiment_test1)
knn_result_20set1 = table(knn.20set1, knn_sentiment_test1)

knn_result_1set2 = table(knn.1set2, knn_sentiment_test2)
knn_result_5set2 = table(knn.5set2, knn_sentiment_test2)
knn_result_20set2 = table(knn.20set2, knn_sentiment_test2)

#pooling the accuracies
knn_table_set1 = data.frame(correc1_set1, correc5_set1, correc20_set1)
colnames(knn_table_set1) = c("Accuracy for k=1", "Accuracy for k=5", "Accuracy for k=20")

knn_table_set2 = data.frame(correc1_set2, correc5_set2, correc20_set2)
colnames(knn_table_set2) = c("Accuracy for k=1", "Accuracy for k=5", "Accuracy for k=20")


########################################################################################################
####################################################
##PART THREE (C): NNET BUILDING ########################################################################
####################################################
########################################################################################################

####################################################
#use knn_set1, knn_set2, since this requires normalized data, fixing the data all over
####################################################

nnet_ds1 = data.frame(knn_set1$senty, knn_set1$score, knn_set1$y, stringsAsFactors = FALSE)
nnet_ds2 = data.frame(knn_set2$tfidf, knn_set2$y, stringsAsFactors = FALSE)

colnames(nnet_ds1) = c("senty", "score", "y")
colnames(nnet_ds2) = c("tfidf", "y")

knx = scale(nnet_ds2$tfidf, center=TRUE,scale=TRUE)
nnet_ds2$tfidf_scaled = knx

nnet_ds1$y_new = 0
for(i in 1:nrow(nnet_ds1)){
  if(as.character(nnet_ds1$y[i])=="hate")
  {
    nnet_ds1$y_new[i] = 1
  }
  if(as.character(nnet_ds1$y[i])=="offensive")
  {
    nnet_ds1$y_new[i] = 0.5
  }
}

nnet_ds2$y_new = 0
for(i in 1:nrow(nnet_ds2)){
  if(as.character(nnet_ds2$y[i])=="hate")
  {
    nnet_ds2$y_new[i] = 1
  }
  if(as.character(nnet_ds2$y[i])=="offensive")
  {
    nnet_ds2$y_new[i] = 0.5
  }
}

nnet_ds1$y =
  nnet_ds2$y = 
  nnet_ds2$tfidf = NULL

ideal_set1 = class.ind(nnet_ds1$y_new)
ideal_set2 = class.ind(nnet_ds2$y_new)


####################################################
#actual nnet part
ind = 1:1000

####################################################
#using everything but the last column, model towards the last column, then pool into results table
####################################################

my_ann_set1 = nnet(nnet_ds1[-ind,-3], ideal_set1[-ind,], size = 10, softmax = TRUE)

ann1_result = table(predict(my_ann_set1, nnet_ds1[ind,-3], type="class"), nnet_ds1[ind,]$y_new)

my_ann_set2 = nnet(nnet_ds2[-ind,-2], ideal_set2[-ind,], size = 10, softmax = TRUE)

ann2_result = table(predict(my_ann_set2, nnet_ds2[ind,-2], type = "class"),nnet_ds2[ind,]$y_new)


########################################################################################################
####################################################
##PART FOUR: 10FOLD CROSS VALIDATION####################################################################
####################################################
########################################################################################################

####################################################
#10fold fold building up
####################################################

K = 10

####################################################
# createFolds for speech sentiment to predict
####################################################

Folds = cut(seq(1, nrow(mydf_v3_2)), breaks = K, labels = FALSE)

fold_set1 = data.frame(data_me1, stringsAsFactors = FALSE)
fold_set2 = data.frame(data_me2, stringsAsFactors = FALSE)

colnames(fold_set1) = c("ID","text","sentiment","score","y")
colnames(fold_set2) = c("ID","text", "tfidf","y")

#set 1 svm error rate calculating
cv10_svm_set1 = sapply(1:K, FUN = function(i){
  testID = which(Folds == i, arr.ind = TRUE)
  test = fold_set1[testID,]
  train = fold_set1[-testID,]
  svm = svm(y~sentiment + score, data = train, kernel = "radial", gamma = 3, cost = 10) #remember the best cost and gamma use
  svm.pred = predict(svm, test)
  svm.pred = data.frame(svm.pred)
  estimate_svm = mean(svm.pred[,1] != test$y)
  return(estimate_svm)
})
err_svm_set1 = mean(cv10_svm_set1) #gets error rate over the 10 fold calculations

#set 2 svm error rate calculating
cv10_svm_set2 = sapply(1:K, FUN = function(i){
  testID = which(Folds == i, arr.ind = TRUE)
  test = fold_set2[testID,]
  train = fold_set2[-testID,]
  svm = svm(y~tfidf, data = train, kernel = "radial", gamma = 0.5, cost = 0.1) #remember the best cost and gamma use
  svm.pred = predict(svm, test)
  svm.pred = data.frame(svm.pred)
  estimate_svm = mean(svm.pred[,1] != test$y)
  return(estimate_svm)
})
err_svm_set2 = mean(cv10_svm_set2)

err_svm_df = data.frame(err_svm_set1, err_svm_set2)
colnames(err_svm_df) = c("Error 10CV Set1", "Error 10CV Set2")

#nnet tuning
#online resource just says use tune.nnet

tmodel_set1 = tune.nnet(y_new~senty+score, data = nnet_ds1, size = 1:10)
summary(tmodel_set1)

# best size is 1

tmodel_set2 = tune.nnet(y_new~tfidf_scaled, data = nnet_ds2, size = 1:10)
summary(tmodel_set2)

#best size is 2


#set1 nnet 10cv
# cv10_nnet_set1 = sapply(1:K, FUN = function(i){
#   testID = which(Folds == i, arr.ind = TRUE)
#   test = nnet_ds1[testID,]
#   train = nnet_ds1[-testID,]
#   nnet = nnet(y_new~senty+score, data = nnet_ds1, size = 1) #remember the best cost and gamma use
#   nnet.pred = predict(nnet, test)
#   nnet.pred = data.frame(nnet.pred)
#   estimate_svm = mean(nnet.pred[,1] != test$y_new)
#   return(estimate_svm)
# })
# err_nnet_set1 = mean(cv10_nnet_set1)
# 
# #set1 nnet 10cv
# cv10_nnet_set2 = sapply(1:K, FUN = function(i){
#   testID = which(Folds == i, arr.ind = TRUE)
#   test = nnet_ds2[testID,]
#   train = nnet_ds2[-testID,]
#   nnet = nnet(y_new~tfidf_scaled, data = nnet_ds2, size = 2) #remember the best cost and gamma use
#   nnet.pred = predict(nnet, test)
#   nnet.pred = data.frame(nnet.pred)
#   estimate_svm = mean(nnet.pred[,1] != test$y_new)
#   return(estimate_svm)
# })
# err_nnet_set2 = mean(cv10_nnet_set2)
# 
# err_nnet_df = data.frame(err_nnet_set1, err_nnet_set2)
# colnames(err_nnet_df) = c("Error 10CV Set1", "Error 10CV Set2")

#above returns odd values

####################################################
####################################################
#PLOTTING WITH ROCR AND THE RESULTS
####################################################
####################################################

#7/13/19
############################################
#ROC CURVES
############################################

#The ROCR package can be used to produce ROC curves. First write a short function to plot an ROC curve
#given a vector containing a numerical score for each observation, pred, and a vector containing the class label for each
#observation, truth.

library(ROCR)
rocplot=function(pred,truth,...){ predob=prediction(pred,truth);
perf=performance(predob,"tpr","fpr"); plot(perf,...)}

#In order to obtain the fitted values for a given SVM model fit, we use decision.values=T when fitting svm(). Then
#the predict() function will output the fitted values.

svmfit.opt=svm(y~.,data=dat[train,],kernel="radial",gamma=2,cost=1,decision.values=T)
fitted=attributes(predict(svmfit.opt,dat[train,],decision.values=T))$decision.values

#Now we can produce the ROC plot.

par(mfrow =c(1,2))
rocplot(fitted,dat[train,"y"], main="Training Data")

#SVM appears to be producing accurate predictions. By increasing gamma we can produce a more flexible fit and generate
#further improvements in accuracy.

svmfit.flex=svm(y~.,data=dat[train,],kernel="radial",gamma=50,cost=1,decision.values=T)
fitted=attributes(predict(svmfit.flex,dat[train,],decision.values=T))$decision.values

rocplot(fitted,dat[train,"y"],add=T,col="red")

#These ROC curves are all on the training data. We are really more interested in the level of prediction accuracy on the test
#data.

fitted=attributes(predict(svmfit.opt,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],main="Test Data")

fitted=attributes(predict(svmfit.flex,dat[-train,],decision.values=T))$decision.values
rocplot(fitted,dat[-train,"y"],add=T,col="red"))




####################################################
####################################################
####################################################

#### END OF PROGRAM

####################################################
####################################################
####################################################