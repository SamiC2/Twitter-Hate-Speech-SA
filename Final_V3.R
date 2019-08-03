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
library(e1071)

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

afinn_list = get_sentiments("afinn")

####################################################
##function to get word score
####################################################

get_word_score = function(word){
  word = str_to_lower(word)
  if(word %in% afinn_list$word){
    return (afinn_list$score[afinn_list$word==word])
  }
  return (NULL)
}

####################################################
##function to get sentence score
####################################################

get_sentence_score = function(sentence){
  sentence_score = 0
  words = strsplit(removePunctuation(as.character(sentence)), " ")[[1]]
  
  for(word in words)
  {
    word_score = get_word_score(word)
    if(!is.null(word_score))
    {
      sentence_score = sentence_score + word_score
    }
  }
  return (sentence_score)
}

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

freq = clean_tweets %>% group_by(tweet_id) %>% mutate(word = str_extract(word, "[a-z']+")) %>% count(word, sort = T) %>% ungroup() %>% mutate(probability = n / sum(n)) %>%  select(-n) %>% spread(tweet_id, probability)


####################################################
##replace NAs with 0
####################################################

freq[is.na(freq)] = 0

####################################################
##apply shannon.entropy function to the frequencies/probabilities
####################################################

freq = freq %>% group_by(tweet_id) %>% mutate(entropyHate = shannon.entropy(hate))
freq = freq %>% group_by(tweet_id) %>% mutate(entropyOffensive = shannon.entropy(offensive))
freq = freq %>% group_by(tweet_id) %>% mutate(entropyRegular = shannon.entropy(regular))

freq_2 = unique(freq %>% group_by(tweet_id))
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

####################################################
## PART THREE: SVM AND 10FOLD ANALYSIS
####################################################

#set1 = data.frame(mydf_v3_1, mydf_v3_2) = x1
#set2 = data.frame(tfidf_tweets) = x2
#load the inputs, these should be matrices
x = (0) #feature set i want to train, so I would use one x for sentiment+entropy, and one for tfidf

#the output is the thing I am comparing against, the speech
y = mydf_v3_2$speech

#put the data together into one dataframe

data_me = data.frame(x=x, y=as.factor(y))

#split the data into test and train
#unsure

train = sample(200,100) #what exactly needs to be here

#fit with svm function

svm.poly = svm(y~., data = data_me[train,], kernel = "polynomial", gamma = 1, cost = 1)

# #Find the best gamma and cost:

tune.out = tune(svm,y~.,data = data_me[train,],kernel = "polynomial", ranges = list(cost = 10^(-1:2), gamma = c(0.5, 1:4)))

summary(tune.out)

ypred = predict(tune.out$best.model, newdata = data_me[-train,])

table(predict = ypred, truth = data_me[-train,"y"])

#fit training data to svm function
svm.poly = svm(y~.,data=data_me[train,],kernel = "polynomial", degree=2, cost=1)

tune.out=tune(svm,y~.,data=data_me[train,],kernel="polynomial",ranges=list(cost=10^(-1:2),degree=(1:4)))
summary(tune.out)
ypred=predict(tune.out$best.model,newdata=data_me[-train,])
table(predict=ypred,truth=data_me[-train,"y"])


####################################################
## PART FOUR: CROSS VALIDATION
####################################################

