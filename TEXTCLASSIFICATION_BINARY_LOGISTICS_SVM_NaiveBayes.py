#Theroy, we have two sets of data. True or legit data and False or misinformation data.
#We are using scikit learn libraries to perform classification
#We are using three Algorithms Naive Bayes, SVM and Logistic regression with accuracy metirc & confustion matrix.
#We split the data and create a pipeline with (Vectorizer to vectorize the data) see the below example
'''from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())
'''
#tfidf_term_frequency transformer and the actual algorithms are chosen
#We fit the training data to model to train 
##Then we predict the model
#Generate accuracy report or confustion matrix with target and predicted

#Key last note ,, if you wonder how we preprocessed the data its using this library vectorizer...  It removes punctuation by default, underscores and removes stopwords, perform lemmetaziation
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 13:12:43 2021

@author: Gabriel@Asus
"""

#Fake News Classifier

#Get data
#C:\Users\Asus\Desktop\PersonalResearch\NLP\FakeDataForML



#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import  LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

#Reading CSV files
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

#Lets set target label
fake['target'] = 'fake'
true['target'] = 'true'

#Now we have to concat two tables
news = pd.concat([fake, true]).reset_index(drop=True)
news.head()
#CountVectorizer removes punctuation and makes the data in machine readable format

#Lets  split the data
#Train-test split
x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=1)


#Data Exploration and Data Engineering

#Logistic regression classification


LogisticPipeline = Pipeline(
    [
     ('vectorizer',CountVectorizer()),
     ('tfidf_term_frequency',TfidfTransformer()),
     ('Logistic regression model', LogisticRegression())
     ]
    )


model_logsticsregression = LogisticPipeline.fit(x_train,y_train)
LogisticsRegression_model_predictor = model_logsticsregression.predict(x_test)

#Print metricts
print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(y_test, LogisticsRegression_model_predictor)*100,2)))
print("\nConfusion Matrix of Logistic Regression Classifier:\n")
print(confusion_matrix(y_test, LogisticsRegression_model_predictor))
print("\nCLassification Report of Logistic Regression Classifier:\n")
print(classification_report(y_test, LogisticsRegression_model_predictor))


#Support Vector classification
SVMPipeline = Pipeline([('vectorizer', CountVectorizer()), ('tfidfTransformer', TfidfTransformer()), ('SVCmodel', LinearSVC())])

model_svc = SVMPipeline.fit(x_train, y_train)
svc_predictor = model_svc.predict(x_test)

print("Accuracy of SVM Classifier: {}%".format(round(accuracy_score(y_test, svc_predictor)*100,2)))
print("\nConfusion Matrix of SVM Classifier:\n")
print(confusion_matrix(y_test, svc_predictor))
print("\nClassification Report of SVM Classifier:\n")
print(classification_report(y_test, svc_predictor))

##Naive-Bayes classification
NaivesBayesPipeline = Pipeline([('vectorizer', CountVectorizer()), ('tfidfTransformer', TfidfTransformer()), ('NaivesBayesmodel', MultinomialNB())])
model_nb = NaivesBayesPipeline.fit(x_train, y_train)
naivesbayes_predictor = model_nb.predict(x_test)

print("Accuracy of Naive Bayes Classifier: {}%".format(round(accuracy_score(y_test, naivesbayes_predictor)*100,2)))
print("\nConfusion Matrix of Naive Bayes Classifier:\n")
print(confusion_matrix(y_test, naivesbayes_predictor))
print("\nClassification Report of Naive Bayes Classifier:\n")
print(classification_report(y_test, naivesbayes_predictor))



