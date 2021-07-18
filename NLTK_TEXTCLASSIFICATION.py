# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:29:50 2021

@author: Asus
"""

#TextClassification with NLTK

import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegression
#Pipeline 
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

np.random.seed(500)
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
from sklearn.model_selection import train_test_split

#Lets  split the data

# Step - a : Remove blank rows if any.
news['text'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
news['text'] = [entry.lower() for entry in news['text']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
news['text']= [word_tokenize(entry) for entry in news['text']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(news['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    news.loc[index,'text_final'] = str(Final_words)
    
    
    
    
#Train-test split

x_train,x_test,y_train,y_test = train_test_split(news['text'], news.target, test_size=0.2, random_state=1)


Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

LogisticPipeline = Pipeline(
    [
     ('TFidFvectorizer',TfidfVectorizer()),
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
SVMPipeline = Pipeline([('TFidFvectorizer', TfidfVectorizer()), ('tfidf_term_frequency', TfidfTransformer()), ('SVCmodel', LinearSVC())])

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



