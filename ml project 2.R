library(keras)
library(dplyr)
library(ggplot2)
library(purrr)

IMDB=dataset_imdb(num_words = 10000)
train=IMDB$train
test=IMDB$test
train_x=train$x
train_y=train$y
test_x=test$x
test_y=test$y

word_index=dataset_imdb_word_index()
word_index_df=data.frame(
  word = names(word_index),
  idx = unlist(word_index, use.names = FALSE),
  stringsAsFactors = FALSE
)

# The first indices are reserved  
word_index_df <- word_index_df %>% mutate(idx = idx + 3)
word_index_df <- word_index_df %>%
  add_row(word = "<PAD>", idx = 0)%>%
  add_row(word = "<START>", idx = 1)%>%
  add_row(word = "<UNK>", idx = 2)%>%
  add_row(word = "<UNUSED>", idx = 3)

word_index_df <- word_index_df %>% arrange(idx)


train_x <- pad_sequences(
  train_x,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

test_x <- pad_sequences(
  test_x,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

#bagging and random forest
library(randomForest)
train_rf_y=as.factor(train_y)
test_rf_y=as.factor(test_y)
bagging=randomForest(train_x,train_rf_y)
bagging
predict_bagging=predict(bagging,newdata=test_x)
predict_bagging
pb=table(predict_bagging,test_rf_y)
sum(diag(pb)/sum(pb))
rf=randomForest(train_x,train_rf_y,ntree = 1000)
rf
predict_rf=predict(rf,newdata=test_x)
predict_rf
prf=table(predict_rf,test_rf_y)
sum(diag(prf)/sum(prf))
plot(margin(rf,test_rf_y))


# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size <- 10000

#GAP1
model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units =16 , activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

x_validation <- train_x[1:10000, ]
partial_x_train <- train_x[10001:nrow(train_x), ]

y_validation <- train_y[1:10000]
partial_y_train <- train_y[10001:length(train_y)]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 40,
  batch_size = 512,
  validation_data = list(x_validation, y_validation),
  verbose=1
)

scores <- model %>% evaluate(test_x, test_y)
scores

#rnn
model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_simple_rnn(units = 16) %>%
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 40,
  batch_size = 512,
  validation_data = list(x_validation, y_validation),
  verbose=1
)

scores <- model %>% evaluate(test_x, test_y)
scores

#lstm
model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_lstm(units = 16, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 16 , activation = "relu") %>%
  #layer_dropout(0.2)
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 40,
  batch_size = 512,
  validation_data = list(x_validation, y_validation),
  verbose=1
)

scores <- model %>% evaluate(test_x, test_y)
scores




#logistic regression
#glm.fit=glm(train_labels~train_data,data = train,family = binomial)

#lda and qda
#library(MASS)
#lda.fit=lda(train_labels~train_data,data=IMDB,subset=train)
#lda.fit
#qda.fit=qda(train_labels~train_data,data=IMDB,subset=train)
#qda.fit

#boosting
#library(gbm)
#train_labels=as.matrix(train_labels,ncol=256)
#boosting=gbm(train_labels~train_data,distribution='gaussian',n.trees=5000)
#summary(boosting)
#predict_boosting=predict(boosting,newdata=test_labels)