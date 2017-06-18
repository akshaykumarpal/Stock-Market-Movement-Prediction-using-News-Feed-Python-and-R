---
  title: "Stock Market Analysis"
output:
  html_notebook: default
pdf_document: default
---
  
  
  Loading packages
```{r}
library(twitteR)
library(ROAuth)
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
```


Building a model to do sentiment analysis on tweets:
  
  
  Preparing data for model building:
  
  ```{r}

# function for converting some symbols
conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")

##### loading classified tweets ######
# source: http://help.sentiment140.com/for-students/
tweets_classified <- read_csv('/home/blacksaint/Desktop/Farid-ml/classified_tweets/training.csv',
                              col_names = c('sentiment', 'id', 'query','text')) %>%
  
  
  
  # converting some symbols
  dmap_at('text', conv_fun)

# replacing class values
mutate(sentiment = ifelse(sentiment == 0, 0, 1))



# Removing not required columns
tweets_new<-list()
tweets_new$sentiment <-tweets_classified$sentiment
tweets_new$text <-tweets_classified$text
tweets_new$id <- tweets_classified$id
tweets_classified<- tweets_new


# data splitting on train and test
set.seed(2340)
trainIndex <- createDataPartition(tweets_classified$sentiment, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)

tweets_train <- tweets_classified[trainIndex, ]
tweets_test <- tweets_classified[-trainIndex, ]


```

Building and training model:
  ```{r}
# define preprocessing function and tokenization function

prep_fun <- tolower
tok_fun <- word_tokenizer
it_train <- itoken(tweets_train$sentiment, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   ids = tweets_train$id,
                   progressbar = TRUE)
it_test <- itoken(tweets_test$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  ids = tweets_test$id,
                  progressbar = TRUE)

# creating vocabulary and document-term matrix
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, vectorizer)
dtm_test <- create_dtm(it_test, vectorizer)

# define tf-idf model
tfidf <- TfIdf$new()

# fit the model to the train data and transform it with the fitted model
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- fit_transform(dtm_test, tfidf)

# train the model
t1 <- Sys.time()
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = tweets_train[['sentiment']], 
                               family = 'binomial', 
                               # L1 penalty
                               alpha = 1,
                               # interested in the area under ROC curve
                               type.measure = "auc",
                               # 5-fold cross-validation
                               nfolds = 5,
                               # high value is less accurate, but has faster training
                               thresh = 1e-3,
                               # again lower number of iterations for faster training
                               maxit = 1e3)

print(difftime(Sys.time(), t1, units = 'mins'))

print(paste("max AUC on TRAIN =", round(max(glmnet_classifier$cvm), 4)))

preds <- predict(glmnet_classifier, dtm_train_tfidf, type = 'response')[ ,1]
glmnet:::auc(as.numeric(tweets_test$sentiment), preds)

# save the model for future using
saveRDS(glmnet_classifier, '/home/blacksaint/Desktop/Farid-ml/glmnet_classifier.RDS')
```

Performing sentiment analysis on topics:
  ```{r}

# reading tweets from dataset
olc<-read.csv("olc.csv")


# preprocessing and tokenization
it_olc <- itoken(olc$Topic,
                 preprocessor = prep_fun,
                 tokenizer = tok_fun,
                 ids = olc$id,
                 progressbar = TRUE)


# creating vocabulary and document-term matrix
dtm_train <- create_dtm(it_olc, vectorizer)

# transforming data with tf-idf
dtm_tfidf <- fit_transform(dtm_train, tfidf)

# loading classification model
glmnet_classifier <- readRDS('/home/blacksaint/Desktop/Farid-ml/glmnet_classifier.RDS')

# predict probabilities of positiveness
preds<- predict(glmnet_classifier, dtm_tfidf, type = 'response')[ ,1]

# adding sentiment rates to initial dataset
olc$sentiment <- preds

write.csv(olc,'DJ.csv')


```

Preparing data for stock analysis:
  
  ```{r}

#reading stored csv file
DJ<-read.csv("DJ.csv")

#removing unwanted columns
DJTrain$X.1<-NULL
DJTrain$X<-NULL

#Scaling data
maxs=apply(DJ[,c("DJIA","Volume")],2,max)
mins=apply(DJ[,c("DJIA","Volume")],2,min)

DJ[,c("DJIA","Volume")]<-
  as.data.frame(scale(bank[,c("DJIA","Volume")], 
                      center=mins, scale=maxs-mins))


#Dividing data into Train and Test
div = 1:floor(nrow(DJ)*0.9)
DJTrain=DJ[div,]
DJTest=DJ[-div,]


head(DJTrain)
View(DJTrai)
plot(DJTrain$DJIA, type = 'l')
lines(DJTrain$sentiment, type='l',col='green')
lines(DJTrain$Volume, type='l',col='red')
```

LINEAR REGRESSION:
  
  ```{r}

#Build linear models first: 
lmDJ=lm(DJIA~Volume+sentiment, data=DJTrain)
anova(lmDJ)


#Predicting values
predlmDJTrain=predict(lmDJ, newdata=DJTrain)
predlmDJTest=predict(lmDJ, newdata=DJTest)


mseTrain.lmDJ=sum((predlmDJTrain-DJTrain$DJIA)^2)
mseTest.lmDJ=sum((predlmDJTest-DJTest$DJIA)^2)
print(mseTrain.lmDJ)
print(mseTest.lmDJ)
```

POLYNOMIAL REGRESSION:
  
  ```{r}

VolSent<- (DJTrain$Volume^2 + DJTrain$sentiment^2)
Volcu<- VolSent^2

lmDJ2=lm(DJIA~VolSent+Volcu, data=DJTrain)
anova(lmDJ2)

#Predicting values
predlmDJ2Train=predict(lmDJ2, newdata=DJTrain)
predlmDJ2Test=predict(lmDJ2, newdata=DJTest)


mseTrain.lmDJ2=sum((predlmDJ2Train-DJTrain$DJIA)^2)
mseTest.lmDJ2=sum((predlmDJ2Test-DJTest$DJIA)^2)
print(mseTrain.lmDJ2)
print(mseTest.lmDJ2)


```

LOGISTIC REGRESSION:
  
  ```{r}
#converting stock values in binomial
#when stock value increases = 1 if it decreases = 0
DJeaTrain<-DJTrain
for (a in 2:length(DJTrain$DJIA)){
  if(DJTrain$DJIA[a]> DJTrain$DJIA[a-1]){DJeaTrain$DJIA[a] = 1}else{DJeaTrain$DJIA[a] = 0 }}
DJeaTrain$DJIA[1]<-0

DJeaTest<-DJTest
for (a in 2:length(DJTest$DJIA)){
  if(DJTest$DJIA[a]> DJTest$DJIA[a-1]){DJeaTest$DJIA[a] = 1}else{DJeaTest$DJIA[a] = 0 }}
DJeaTest$DJIA[1]<- 0

#checking for missing value
library(Amelia)
missmap(DJTrain, main = "Missing values vs observed")


logitModel1 <- glm(DJIA ~ sentiment+Volume, family=binomial, data=DJTrain)

# checking significance of variables
summary(logitModel1)
anova(logitModel1, test="Chisq")

library(pscl)
pR2(logitModel1)


#Accessing the predictive ability of the model
fitted.results <- predict(logitModel1,DJTest,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != DJeaTest$Label)
print(paste('Accuracy',1-misClasificError))
```

NEURAL NETWORK(Feed forward MLP):
  ```{r}

library("neuralnet")

#reading saved values
DJTrain<-read.csv('DJTrain.csv')
DJTest<-read.csv('DJTest.csv')

#selecting layers
layers=c(3,3,2)

nnDJ<-neuralnet(DJIA~sentiment+Volume,hidden=layers, 
                linear.output=T, err.fct="sse",data=DJTrain)

plot(nnDJ)
PredDJTrain=neuralnet::compute(nnDJ, DJTrain[,c("sentiment","Volume")])
PredDJTest=neuralnet::compute(nnDJ, DJTest[,c("sentiment","Volume")])

predDJResultTrain=PredDJTrain$net.result
predDJResultTest=PredDJTest$net.result


mse1Train.nn=sum((predDJResultTrain-DJTrain$DJIA)^2) 
mse1Test.nn=sum((predDJResultTest-DJTest$DJIA)^2) 

print(paste("For  Train MSE(LM)=",
            mseTrain.lmDJ, "Train MSE(NN)= ", mse1Train.nn))
print(paste("For  Test MSE(LM)=",
            mseTest.lmDJ, "Test MSE(NN)= ", mse1Test.nn))
```

NEURAL NETWORK with increased number of neuronsn in first hidden layers:
  
  ```{r}

#selecting layers
layers=c(6)

nnDJ3<-neuralnet(DJIA~sentiment+Volume,hidden=layers, 
                 linear.output=T, err.fct="sse",data=DJTrain)

plot(nnDJ3)
PredDJ3Train=neuralnet::compute(nnDJ3, DJTrain[,c("sentiment","Volume")])
PredDJ3Test=neuralnet::compute(nnDJ3, DJTest[,c("sentiment","Volume")])

predDJ3ResultTrain=PredDJ3Train$net.result
predDJ3ResultTest=PredDJ3Test$net.result


mse3Train.nn=sum((predDJ3ResultTrain-DJTrain$DJIA)^2) 
mse3Test.nn=sum((predDJ3ResultTest-DJTest$DJIA)^2) 

lines(predDJ3ResultTrain, col="blue")

print(paste("For  Train MSE(LM)=",
            mseTrain.lmDJ, "Train MSE(NN)= ", mse3Train.nn))
print(paste("For  Test MSE(LM)=",
            mseTest.lmDJ, "Test MSE(NN)= ", mse3Test.nn))



```

Neural Network with BakPropogation:
  
  ```{r}
layers=c(3,2)


nnDJ2 <- neuralnet(DJIA~sentiment+Volume, data=DJTrain, hidden=layers, err.fct="sse",linear.output=FALSE,algorithm="backprop",learningrate=0.1)
plot(nnDJ2)

PredDJ2Train=neuralnet::compute(nnDJ2, DJTrain[,c("sentiment","Volume")])
PredDJ2Test=neuralnet::compute(nnDJ2, DJTest[,c("sentiment","Volume")])

predDJResultTrain2=PredDJTrain2$net.result
predDJResultTest2=PredDJTest2$net.result

mse2Train.nn=sum((predDJResultTrain2-DJTrain$DJIA)^2) 
mse2Test.nn=sum((predDJResultTest2-DJTest$DJIA)^2) 



print(paste("For  Train MSE(LM)=",
            mseTrain.lmDJ, "Train MSE(NN)= ", mse2Train.nn, "Train 1 MSE(NN)= ", mse1Train.nn))
print(paste("For  Test MSE(LM)=",
            mseTest.lmDJ, "Test MSE(NN)= ", mse2Test.nn),"Train 1 MSE(NN)= ", mse1Test.nn)

```

COMPARING MSE OF ALL NN:
  ```{r}
print(paste("For  1st Neural Network Train MSE(NN1)=",
            mse1Train.nn, " 2nd Neural Network - Train MSE(NN2)= ", mse3Train.nn, "3rd Neural Network Train 1 MSE(NN3)= ", mse2Train.nn))

print(paste("For  1st Neural Network Test MSE(NN1)=",
            mse1Test.nn, " 2nd Neural Network - Test MSE(NN2)= ", mse3Test.nn ,"3rd Neural Network Test 1 MSE(NN3)=  " , mse2Test.nn))

```
