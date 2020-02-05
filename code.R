
# load all required libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")

# load data
data(brca)
write.csv(brca, file = "data/brca.csv")

# Explore the data
unique(brca$y)
table(brca$y)

dim(brca$x)
head(brca$x)
colMeans(brca$x)
  
# split the data into training and test sets
set.seed(1)
test_index <- createDataPartition(y =  brca$y, times = 1, p = 0.2, list = FALSE)

y <- droplevels(brca$y[-test_index])
x <- brca$x[-test_index, ]
brca_train_set <- data.frame(x,y)
table(brca_train_set$y)
write.csv(brca_train_set, file = "data/brca_train_set.csv")


y2 <- droplevels(brca$y[test_index])
x2 <- brca$x[test_index, ]
brca_test_set <- data.frame(x2,y2)
table(brca_test_set$y2)
write.csv(brca_test_set, file = "data/brca_test_set.csv")

#############################
############################

# Fit LDA model and measure Accuracy
train_lda <- train(x, y, method = "lda")
train_lda$finalModel$means
train_lda$results["Accuracy"]
train_lda$finalModel

# Which features appear to be driving the algorithm?
t(train_lda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(B, M, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()

colMeans(x)

d <- apply(train_lda$finalModel$means, 2, diff)
ind <- order(abs(d), decreasing = TRUE)[1:2]
plot(x[, ind], col = y)

############################

# Fit LDA model and measure Accuracy with scaling (preProcess set to center)
train_lda2 <- train(x, y, method = "lda", preProcess = "center")
train_lda2$finalModel$means
train_lda2$results["Accuracy"]
train_lda2$finalModel

# Which features appear to be driving the algorithm?
t(train_lda2$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(B, M, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()

colMeans(x)

d <- apply(train_lda2$finalModel$means, 2, diff)
ind <- order(abs(d), decreasing = TRUE)[1:2]
plot(x[, ind], col = y)

############################

# Fit QDA model and measure Accuracy
train_qda <- train(x, y, method = "qda")
train_qda$results["Accuracy"]
train_qda$finalModel

t(train_qda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(B, M, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()

colMeans(x)

d <- apply(train_qda$finalModel$means, 2, diff)
ind <- order(abs(d), decreasing = TRUE)[1:2]
plot(x[, ind], col = y)

#############################
############################

# Apply model ensemble

models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")

fits <- lapply(models, function(model){ 
  print(model)
  train(y ~ ., method = model, data = brca_train_set)
}) 

names(fits) <- models

# Create a matrix of predictions for the test set
pred <- sapply(fits, function(object) 
  predict(object, newdata = brca_test_set))

dim(pred)

acc <- colMeans(pred == brca_test_set$y2)
acc

# dataframe to store accuracy of the different models
model_result <- data.frame(METHOD = models, ACCURACY = acc)
model_result

# ensemble average
mean(acc)
model_result <- bind_rows(model_result, data_frame(METHOD="ensemble average", ACCURACY = mean(acc)))

# build an ensemble prediction by majority vote and compute the accuracy of the ensemble.
votes <- rowMeans(pred == "M")
y_hat <- ifelse(votes > 0.5, "M", "B")
mean(y_hat == brca_test_set$y2)
model_result <- bind_rows(model_result, data_frame(METHOD="ensemble majority vote", ACCURACY = mean(y_hat == brca_test_set$y2)))

model_result %>% knitr::kable()

# Which individual methods perform better than the ensemble?
ind <- acc > mean(y_hat == brca_test_set$y2)
sum(ind)
models[ind]