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


####################################################
##Plotting of SVM results
####################################################




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
##PART FOUR: 10FOLD CROSS VALIDATION####################################################################
####################################################
########################################################################################################

####################################################
#10fold fold building up
####################################################



####################################################
####################################################
#PLOTTING WITH ROCR AND THE RESULTS
####################################################
####################################################






####################################################
####################################################
####################################################

#### END OF PROGRAM

####################################################
####################################################
####################################################