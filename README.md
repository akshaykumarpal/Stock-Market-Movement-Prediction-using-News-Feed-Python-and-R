# Stock Market Movement Prediction using News-Feed in Python and R programming

Environment: Python 3.5.2, Anaconda Navigator (1.3.1), R (3.3.1) and R Studio (1.0.136)

We had taken news feeds from different sources and build a model on a scale of 1 to 4 using sentiment analysis. Also, we took stock data from Dow Jones investors and made the data scaled between 0 to 1 so that we can apply Neural Network. Then, combining these data our final dataset was from past 2008 to 2016 years containing 25 topics for each month. Now we also applied logistic regression to make an indexing of the class label where our objective becomes to predict whether the stock price increases or decreases as ‘0’ and ‘1’ respectively. 

We have applied both Linear Regression and Polynomial Regression to find relation between the change in stock market price vs. the volume as well as sentiment of news articles. We applied our training data on two classification methods such as Logistic Regression and Multi-Layer Perceptron (MLP) to estimate the movement of change in stock market price, volume and the sentiments of news articles. We trained our model on the basis of TF-IDF (Term Frequency – Inverse Document Frequency) sparse matrix using both unigram and bigram scores.

We saw Bigram model performs better than Unigram model for both Multi-Layer Perceptron and Logistic Regression. Also, we found that Multi-Layer Perceptron gives us the better accuracy than Logistic Regression.
