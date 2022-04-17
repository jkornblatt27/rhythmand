import logging
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from helpers import score_report, make_bow

logger = logging.getLogger(__name__)


def naive_bayes_model(features, labels, text_type, n=1, alpha=1.0):
    """Runs a multinomial Naive Bayes model with user-specified text processing type, n-gram and alpha parameters"""

    X = features
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1227)

    # Create tfidf features and train the model
    logger.info('Generating features')
    vectorizer, features, tfidf, train_tfidf_features = make_bow(X_train, n)
    logger.info('Feature generation complete. Fitting model')
    model = MultinomialNB(alpha=alpha)
    model.fit(train_tfidf_features, y_train)

    # Generate tfidf features for the test set and make predictions
    test_data_features = vectorizer.transform(X_test)
    test_tfidf_features = tfidf.fit_transform(test_data_features)

    preds = model.predict(test_tfidf_features)

    # Get the score report
    score_report(y_test, preds, model_type='naive_bayes', text_type=text_type, n=n, alpha=alpha)

    # Save model and vectorizer
    countvectfile = str(text_type) + 'naive_bayes_' + str(n) + '_gram_' + str(alpha) + '_alpha_countvectorizer.pickle'
    tfidffile = str(text_type) + 'naive_bayes_' + str(n) + '_gram_' + str(alpha) + '_alpha_tfidfvectorizer.pickle'
    modelfile = str(text_type) + 'naive_bayes_' + str(n) + '_gram_' + str(alpha) + '_alpha_model.pickle'
    pickle.dump(vectorizer, open(countvectfile, 'wb'))
    pickle.dump(tfidf, open(tfidffile, 'wb'))
    pickle.dump(model, open(modelfile, 'wb'))


if __name__ == '__main__':
    data = pd.read_csv("emotions_clean.csv", encoding='utf-8')

    text_types = ['text', 'clean_no_stops', 'clean_lemmatized', 'all_processing']

    # Iterate through all different text cleaning types
    for text_type in text_types:
        text = data[text_type].tolist()
        labels = data['label'].tolist()

        naive_bayes_model(text, labels, text_type, 1, 1)  # Unigrams and alpha=1
        naive_bayes_model(text, labels, text_type, 2, 1)  # Unigrams and bigrams and alpha=1
        naive_bayes_model(text, labels, text_type, 1, 0)  # Unigrams and alpha=0
        naive_bayes_model(text, labels, text_type, 2, 0)  # Unigrams and bigrams and alpha=0
