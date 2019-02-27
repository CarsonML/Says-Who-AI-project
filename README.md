# Says-Who-AI-project
Goal: to use both an SVM and Naive-Bayes model to differentiate between lyrics by The Beatles and Taylor Swift


This project's main goal was to try to differentiate between lyrics by the two different artists mentioned above.

# Preprocessing
The first step was to preprocess the data. First, I made all lyrics lowercase and removed punctuation, as well as removing all  oddly short or long data. Then I removed duplicate lyrics from the dataset. Since there were still some lines that were nonsensical, I also removed all data that started with two characters that weren't letters or numbers.

# Working with Data
The data is in the form of a text file. I split the data into X and Y sets, then stemmed all words in an attemt to increase accuracy. To finalize the data structure, I used scikitlearn to both do a train/test split, and vectorize the data

# Creating Model
The next step was to create the model. I split the previous data into 10 sets in order to implement 10-fold cross validation, and thus get a more accurate estimate for the model accuracy. I created two functions, one that created a support vector machine (SVM) and one that created a Naive-Bayes model, that would implement this cross validation, and return the highest performing model of the given type and its accuracy.

# Implementation
I called both model-generation functions, determined which model had a better accuracy, trained that model on the entire training dataset, and then returned the overall accuracy.

Warning: Final accuracy was not great, about 55% or so if I remember correctly.


Made while working with Hello World Studio
