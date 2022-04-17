import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def score_report(true, predicted, model_type='naive_bayes', text_type='all_processing', n=1,
                 alpha=None, c=None):
    """Prints the accuracy, f1, precision, and recall scores, and writes to a text file"""
    accuracy = accuracy_score(true, predicted)
    f1 = f1_score(true, predicted, average='weighted')
    precision = precision_score(true, predicted, average='weighted')
    recall = recall_score(true, predicted, average='weighted')

    print('Accuracy: %s' % accuracy)
    print('F1 Score: %s' % f1)
    print('Precision: %s' % precision)
    print('Recall: %s' % recall)

    filename = str(model_type) + "_results.txt"
    textfile = open(filename, "a")

    if model_type == 'naive_bayes':
        textfile.write(
            'Results for naive_bayes with ' + str(text_type) + ' and ' + str(n) + '-gram BOW and alpha=' + str(
                alpha) + ":")
    elif model_type == 'svm':
        textfile.write('Results for svm with ' + str(text_type) + ' and ' + str(n) + '-gram BOW and c=' + str(c) + ":")

    textfile.write("\n")
    textfile.write("\n")
    textfile.write("Accuracy: %s" % accuracy)
    textfile.write("\n")
    textfile.write("F1 Score: %s" % f1)
    textfile.write("\n")
    textfile.write("Precision: %s" % precision)
    textfile.write("\n")
    textfile.write("Recall: %s" % recall)
    textfile.write("\n")
    textfile.write("\n")
    textfile.close()


def make_bow(clean_text, n=1):
    """Takes a list of strings and creates tfidf features using user-specified n-gram tokens"""
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 ngram_range=(1, n), max_features=10000, min_df=5, max_df=0.5)
    features = vectorizer.fit_transform(clean_text)
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(features)
    return vectorizer, features, tfidf, tfidf_features
