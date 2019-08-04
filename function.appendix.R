#Sami Chowdhury
#423006267
#8/3/19

#Helper Functions to aid project

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

source('shannon.entropy.R')

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
##function to get Entropy word score
####################################################

get_entropy_word = function(word, freq){
  word = str_to_lower(word)
  if(word %in% freq$word){
    score = shannon.entropy(freq$probability[freq$word==word])
    return (score)
  }
  return (NULL)
}

####################################################
##function to get Entropy Total Score
####################################################

get_entropy_sentence = function(sentence, freq){
  entropy_score = 0
  words = strsplit(removePunctuation(as.character(sentence)), " ")[[1]]
  
  for(word in words)
  {
    word_score = get_entropy_word(word, freq)
    if(!is.null(word_score))
    {
      entropy_score = entropy_score + word_score
    }
  }
  return (entropy_score)
}



