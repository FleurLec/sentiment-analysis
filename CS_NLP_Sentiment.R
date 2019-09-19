pkg <- c("tm","wordcloud", "text2vec", "xgboost", "dplyr")
sapply(pkg, install.packages)

gc(reset = TRUE)


library(tm)
library(wordcloud)
library(text2vec)
library(dplyr)

##########################################
###Case Study: NLP & Sentiment Analysis###
##########################################

##define utility functions##
HTMLParse <- function(html.text)
  { pattern <- "<.*?>|\\\\|'"
    plain.text <- gsub(pattern, "", html.text)
    return(gsub('\"', "", plain.text,fixed = TRUE))
  }

###import the database##
dt.train <- data.table::fread("./data/labeledTrainData.tsv",data.table=FALSE)
dt.test <- data.table::fread("./data/testData.tsv",data.table=FALSE)

###define parameter##
label <- 'sentiment'
feature <- 'review'
id <- 'id'

##clean the comments###
dt.train$review <- sapply(dt.train$review,HTMLParse)
dt.test$review <- sapply(dt.test$review,HTMLParse)
###example 1: word frequency###

###overview##
tm.corpus <- tm::Corpus(tm::VectorSource(dt.train[,feature]))

# Remove whitespace
tm.corpus <- tm::tm_map(tm.corpus, tm::stripWhitespace)
# Case conversion
tm.corpus <- tm::tm_map(tm.corpus, tolower)
# Remove stopwords  
tm.corpus <- tm::tm_map(tm.corpus,function(x) tm::removeWords(x,c(tm::stopwords(kind = 'en'),'the','film','movie','movies','films')))
# Remove punctuation
tm.corpus <- tm::tm_map(tm.corpus, tm::removePunctuation)
# Stemming: for plot the frequency word do not need to stem
# tm.corpus <- tm_map(tm.corpus, function(x) stemDocument(x,language = 'english'))
###transfer to matrix
tm.matrix <- tm::DocumentTermMatrix(tm.corpus)
###high frequency words
gc()

save(tm.matrix, file = "./tm.matrix.Rdata")
gc(reset = TRUE)
load(file = "./tm.matrix.Rdata")
tm.matrix <- as.matrix(tm.matrix)
tm.freq <- colSums()

sort(tm.freq,decreasing=TRUE)

rm(tm.matrix)
wordcloud::wordcloud(names(tm.freq), tm.freq, min.freq=2000, max.words = 75, colors=RColorBrewer::brewer.pal(8, "Dark2"))

barplot(tm.freq[1:20], xlab = "term", ylab = "frequency",  col=cm.colors(50))

###example 2: sentiment analysis###

# devtools::install_github("dselivanov/text2vec")
require(text2vec)
library(dplyr)
train.token <- dt.train[,feature] %>%tolower %>%word_tokenizer

NLP.train <- itoken(train.token,ids = dt.train[,id], progressbar = FALSE)

NLP.dictionary <- create_vocabulary(NLP.train,stopwords = c(tm::stopwords(kind = 'en'),'the','film','movie','movies','films'), ngram = c(1L, 2L))

NLP.dictionary.pruned <- prune_vocabulary(NLP.dictionary, 
                                          term_count_min = 10, 
                                          doc_proportion_max = 0.5)

NLP.vectorizer <- vocab_vectorizer(NLP.dictionary.pruned)
NLP.matrix.train <- create_dtm(NLP.train, NLP.vectorizer)

param_xgb <- list(objective = "binary:logistic",
                  booster = "gbtree",
                  eta = 0.1,
                  max_depth = 3,
                  min_child_weight = 10,
                  subsample = 0.8,
                  colsample_bytree = 1,
                  nthread = 7,eval_metric='auc')

train.xgbCV_ <- xgboost::xgb.cv(data= NLP.matrix.train,
                                label = dt.train[,label],
                                params=param_xgb,
                                nrounds=2000,
                                nfold=5,
                                stratified=F,
                                verbose=1,
                                early_stopping_rounds=100,
                                maximize=T)

xgbMovie <- xgboost::xgboost(data=NLP.matrix.train,
                            label = dt.train[,label],
                            params=param_xgb,
                            nrounds=train.xgbCV_$best_iteration,
                            verbose=TRUE,
                            maximize=TRUE)


save(xgbMovie, file = "./xgbMovie.Rdata")
save(NLP.vectorizer, file = "./dicoMovie.Rdata")



###predict on the test set##
test.token <- dt.test[,feature] %>%tolower %>%word_tokenizer

NLP.test <- itoken(test.token,ids = dt.test[,id], progressbar = FALSE)
NLP.matrix.test <- create_dtm(NLP.test, NLP.vectorizer)
dt.test$sentiment <- predict(xgbMovie, newdata = NLP.matrix.test,type='response')


##AUC 95.523

###predict on the comments of the team##
dt.team <- data.frame(id=NA,review=NA)
dt.team[1,] <- c('colleague1',"Incredible. I was definitely impressed that this movie made so much entries regarding the quality of the actors")
dt.team[2,] <- c('colleague2',"the movie was a bit disappointing. The actors weren't convincing and the movie was way too long")
dt.team[3,] <- c('colleague3','confusing, but exciting. Overlall it was quite liberating')
dt.team[4,] <- c('colleague4','a little fucked up but in the end it was inspiring and mind opening')
dt.team[5,] <- c('colleague5','I didn’t see the moive. But I guess it is awesome.')
dt.team[6,] <- c('colleague6','The movie was interesting: I really enjoyed the ability of the human being to always find solution to unexpected situations')
dt.team[7,] <- c('colleague7','This movie is not as bad as reviews mentioned but the performance of some actors was not good enough to show the complexity of their character. Mixed feelings : )')
dt.team[8,] <- c('colleague8','I hate the fact that i loved the movie. I am so ashamed of liking a movie like this!')
dt.team[9,] <- c('colleague9','Tsukiji Wonderland is an inspiring documentary about Tokyo’s renowned seafood market. It makes you deep dive into an unique microcosm dedicated to fishing, where every stakeholder is committed to deliver excellence to end customers.')


dt.team <- data.frame(id=NA,review=NA)
dt.team[1,] <- c('colleague1',"Incredible. I was definitely impressed that this movie made so much entries regarding the quality of the actors")

team.token <- dt.team[,feature] %>%tolower %>%word_tokenizer

NLP.team <- itoken(team.token, ids = 1, progressbar = FALSE)
NLP.matrix.team <- create_dtm(NLP.team, NLP.vectorizer)
(sentiment <- predict(xgbMod_,newdata = NLP.matrix.team,type='response'))


###Introduce real test set###
dt.answer <- data.table::fread("./data/test_submission.csv",data.table=FALSE)
names(dt.answer)[names(dt.answer)%in%'sentiment'] <- 'truth'

dt.answer <- merge(x=dt.answer,y=dt.test,by='id',all.x=T)
dt.answer$sentiment.cut <- ifelse(dt.answer$sentiment>=0.5,1,0)

caret::confusionMatrix(as.character(dt.answer$truth),as.character(dt.answer$sentiment.cut)) 