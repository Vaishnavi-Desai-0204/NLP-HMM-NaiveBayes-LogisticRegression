CS 491/691 Project 2: Natural Language Processing
This GitHub repository contains the code for Project 2 of the CS 491/691 course on Natural Language Processing. The project focuses on implementing various algorithms and models for POS tagging, Naïve Bayes classification, and logistic regression. The repository includes the necessary files and functions to load and process data, train the models, and evaluate their performance.

Project Structure
The repository has the following structure:

helpers.py: This file contains helper functions for data loading and preprocessing.

load_pos_data(fname): Loads the POS tagging data from the file fname and returns it in a suitable format for further processing. The function allows for preprocessing of the data, such as removing least common words. Details about the preprocessing steps should be documented in the README.txt file.
load_spam_data(dirname): Loads the spam/ham data from the directory dirname, where dirname should have two subdirectories with spam and ham data. The function converts each spam/ham file into a Bag of Words vector and performs preprocessing steps such as removing common and uncommon words. The returned data includes separate training and testing datasets. The division is 80% for training and 20% for testing. Further details about the preprocessing steps should be mentioned in the README.txt file.
hmm.py: This file contains the implementation of the Hidden Markov Models (HMMs) for POS tagging.

train_hmm(train_data): Takes the train data obtained from helpers.load_pos_data(fname) and generates the transition and emission tables for the HMM. The function stores and returns these tables in any suitable format.
test_hmm(test_data, hmm_transition, hmm_emission): Takes the test data obtained from helpers.load_pos_data(fname) along with the transition and emission tables. It applies the Viterbi algorithm to compute the highest probability sequence of tags for each item in the test data. Then, it compares the predicted tags with the actual tag labels in the test data and calculates the average accuracy and per-sequence accuracy. The function returns both the average accuracy and per-sequence accuracy.
naive_bayes.py: This file contains the implementation of the Naïve Bayes algorithm for classification.

train_naive_bayes(train_data): Calculates the class probabilities and conditional word probabilities for each class using the training data. The function trains a Naïve Bayes model and returns this model in a suitable format.
test_naive_bayes(test_data, model): Takes the test data and the trained Naïve Bayes model as input. For each sample in the test data, the function uses the model to predict the most likely class and compares it with the true label. It calculates the overall accuracy and returns it as the only output.
logistic_regression.py: This file contains the implementation of logistic regression for classification.

train_logistic_regression(train_data): Takes the training data and trains a logistic regression model. The trained model is returned in a format of your choice.
test_logistic_regression(test_data, model): Takes the test data and the trained logistic regression model as input. For each sample in the test data, the function predicts the most likely class using the model and compares it with the true label. It calculates the overall accuracy and returns it as the only output.
test_script.py: This script is used to evaluate the implementations of the models. It ensures that the code runs without errors and evaluates the accuracy of the models based on the provided datasets.

README.txt: This file should contain additional information about the project
