library(magrittr)
library(text2vec)
#library("xgboost")

load(file = "./xgbMovie.Rdata")
load(file = "./dicoMovie.Rdata")

review <- "that's excellent"


team.token <- review %>%tolower %>%word_tokenizer

NLP.team <- itoken(team.token, ids = 1, progressbar = FALSE)
NLP.matrix.team <- create_dtm(NLP.team, NLP.vectorizer)
(sentiment <- predict(xgbMovie, newdata = NLP.matrix.team,type='response'))
