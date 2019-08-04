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

set.seed(1)

source('shannon.entropy.R')
source('function.appendix.R')

data(stop_words)

####################################################
##read data into dfs (twice because of diff manips)
####################################################
mydf_v3_1 = readr::read_csv('hate.speech.csv') #for sentiment analysis
mydf_v3_2 = readr::read_csv('hate.speech.csv') #for entropy and tf

####################################################
# PART ONE (A): SENTIMENT ANALYSIS
####################################################


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

####################################################
## PART ONE (B): ENTROPY SECTION
####################################################

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

####################################################
## PART TWO: tf_idf section
####################################################

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

####################################################
## PART THREE: SVM AND 10FOLD ANALYSIS
####################################################


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

set.seed(149)

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
##Find the best gamma and cost:
####################################################
tune.out = tune(svm, y~., data = train1, kernel = "radial", ranges = list(cost = 10^(-1:2), gamma = c(0.5, 1:4)))

####################################################
##Summary state and tuning
####################################################
summary(tune.out)

ypred = predict(tune.out$best.model, newdata = test1)

table(predict = ypred, truth = test1$y)

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

ypred2 = predict(tune.out2$best.model, newdata = test2)

table(predict = ypred, truth = test2$y)

####################################################
##Now using another method of building a model, kNN
####################################################


####################################################
##PART FOUR: 10FOLD CROSS VALIDATION
####################################################

#10fold fold building up
K = 10

# createFolds for speech sentiment to predict
Folds = createFolds(y,k=K)

# Precision, Recall and F1 null vectors for now
pre.yes = 
  rec.yes =
  f1.yes  =
  pre.no  =
  rec.no  =
  f1.no   = NULL

# confusion matrix is initially empty as well

c.matrix = NULL


# For each fold (observation)
# 1. separate training and testing
# 2. train the model
# 3. predict
# 4. collect the performance in the confusion matrix

for(fold in Folds)
{
  # current fold for testing
  # remainder for training
  
  training = train1[fold,]
  testing  = test1[fold,]
  
  # train the tree and fit the model
  
  the.tree = tree(y~., training)
  cv.tree  = cv.tree(the.tree, FUN=prune.misclass)
  index.best  = which.min(cv.tree$dev)
  best.size   = cv.tree$size[index.best]
  pruned.tree = prune.misclass(the.tree,best=best.size)
  
  # test the tree (predict)
  
  pred = predict(pruned.tree, testing, type="class")
  
  # update the confusion matrix
  
  c.matrix = table(pred, testing$y)
  
  # compute the performance metrics (Precision, recall, F1) for each class
  # add to the proper vector that will be used
  # later for consolidation
  
  pre.yes = c(pre.yes, precision(c.matrix, relevant = "Yes"))
  rec.yes = c(pre.yes,    recall(c.matrix, relevant = "Yes"))
  f1.yes  = c(pre.yes,    F_meas(c.matrix, relevant = "Yes"))
  
  pre.no = c(pre.no, precision(c.matrix, relevant = "No"))
  rec.no = c(pre.no,    recall(c.matrix, relevant = "No"))
  f1.no  = c(pre.no,    F_meas(c.matrix, relevant = "No"))
}

# Now let's build the data frame with the results
# we consolidate the results by computing
#  - mean 
#  - confidence interval (error margin)
# for each metric
results1 = data.frame(
  ci(pre.yes), 
  ci(rec.yes), 
  ci(f1.yes),
  ci(pre.no), 
  ci(rec.no), 
  ci(f1.no)
)

results1 = data.frame(t(results),row.names=NULL)

results1 = cbind(
  Class=c(rep("Yes",3),rep("No",3)),
  Metric=rep(c("Precision","Recall","F1"),2),
  results)


# Let's reorder the factor in column Metric
# so they are plotted in the order Precision, Recall, F1
# otherwise it will be the current (aphabetical) order

results1$Metric = factor(results$Metric, levels = c("Precision", "Recall","F1"))



####################################################
##Repeat CV10F for second set of feautres
####################################################

# Precision, Recall and F1 null vectors for now
pre.yes = 
  rec.yes =
  f1.yes  =
  pre.no  =
  rec.no  =
  f1.no   = NULL

# confusion matrix is initially empty as well

c.matrix = NULL

for(fold in Folds)
{
  training = data_me2[-fold,]
  testing  = data_me2[fold,]
  
  # train the tree and fit the model
  
  the.tree = tree(y~., training)
  cv.tree  = cv.tree(the.tree, FUN=prune.misclass)
  index.best  = which.min(cv.tree$dev)
  best.size   = cv.tree$size[index.best]
  pruned.tree = prune.misclass(the.tree,best=best.size)
  
  # test the tree (predict)
  
  pred = predict(pruned.tree, testing, type="class")
  
  # update the confusion matrix
  
  c.matrix = table(pred, testing$y)
  
  # compute the performance metrics (Precision, recall, F1) for each class
  # add to the proper vector that will be used
  # later for consolidation
  
  pre.yes = c(pre.yes, precision(c.matrix, relevant = "Yes"))
  rec.yes = c(pre.yes,    recall(c.matrix, relevant = "Yes"))
  f1.yes  = c(pre.yes,    F_meas(c.matrix, relevant = "Yes"))
  
  pre.no = c(pre.no, precision(c.matrix, relevant = "No"))
  rec.no = c(pre.no,    recall(c.matrix, relevant = "No"))
  f1.no  = c(pre.no,    F_meas(c.matrix, relevant = "No"))
}

# Now let's build the data frame with the results
# we consolidate the results by computing
#  - mean 
#  - confidence interval (error margin)
# for each metric
results2 = data.frame(
  ci(pre.yes), 
  ci(rec.yes), 
  ci(f1.yes),
  ci(pre.no), 
  ci(rec.no), 
  ci(f1.no)
)

results2 = data.frame(t(results),row.names=NULL)

results2 = cbind(
  Class=c(rep("Yes",3),rep("No",3)),
  Metric=rep(c("Precision","Recall","F1"),2),
  results)


# Let's reorder the factor in column Metric
# so they are plotted in the order Precision, Recall, F1
# otherwise it will be the current (aphabetical) order

results2$Metric = factor(results$Metric, levels = c("Precision", "Recall","F1"))









