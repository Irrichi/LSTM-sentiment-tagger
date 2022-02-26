# LSTM-Sentiment-Tagger
LSTM Sentiment Tagger in pytorch lightning trained on IMDB dataset from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Using model
To train model use main.py. Training takes about 15 minutes on PC with GTX 1050 Ti and achives about 87% accouracy on validation dataset.

To use trained model on some review you need to first train the model and then put the review into review variable in file example.py.
The output is probability that the review has positive sentiment.
Examples:
'I love that film! It is very exciting.' -> 0.9552
'I hate that film! It it really boring.' -> 0.0001
'This is just an over hyped, overrated movie that put me to sleep. During the hype before watching this I expected to see something great. I did not. The movie was so boring and so long. It was a long boring movie.' -> 0.0003

# Model architecture
Model consists of 2-layer LSTM followed with 2 fully connected layers with batch norm layers in between. This model had highest accuracy on test dataset when I was doing tests.
It is using 50 dimensional embedding from Glove with disabled training to prevent overfitting.