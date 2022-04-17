import logging
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from helpers import make_bow, score_report

logger = logging.getLogger(__name__)


def svm_model(features, labels, text_type, n=1, c=5):
    """Trains an SVM model with the user-specified features, labels, text processing type, n-grams, and c parameter"""

    X = features
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1227)

    # Create tfidf features and train the model
    logger.info('Generating features')
    vectorizer, features, tfidf, train_tfidf_features = make_bow(X_train, n)
    logger.info('Feature generation complete. Fitting model')
    model = LinearSVC(C=c, max_iter=1000, random_state=1227, multi_class='ovr')
    model.fit(train_tfidf_features, y_train)

    # Generate tfidf features for the test set and make predictions
    test_data_features = vectorizer.transform(X_test)
    test_tfidf_features = tfidf.fit_transform(test_data_features)

    preds = model.predict(test_tfidf_features)

    # Get the score report
    score_report(y_test, preds, model_type='svm', text_type=text_type, n=n, c=c)

    # Save model and vectorizers
    countvectfile = str(text_type) + 'svm_' + str(n) + '_gram_' + str(c) + '_c_countvectorizer.pickle'
    tfidffile = str(text_type) + 'svm_' + str(n) + '_gram_' + str(c) + '_c_tfidfvectorizer.pickle'
    modelfile = str(text_type) + 'svm_' + str(n) + '_gram_' + str(c) + '_c_model.pickle'
    pickle.dump(vectorizer, open(countvectfile, 'wb'))
    pickle.dump(tfidf, open(tfidffile, 'wb'))
    pickle.dump(model, open(modelfile, 'wb'))


if __name__ == '__main__':
    data = pd.read_csv("emotions_clean.csv", encoding='utf-8')

    text_types = ['text', 'clean_no_stops', 'clean_lemmatized', 'all_processing']

    for text_type in text_types:
        text = data[text_type].tolist()
        labels = data['label'].tolist()

        svm_model(text, labels, text_type, 1, 5)
        svm_model(text, labels, text_type, 2, 5)
        svm_model(text, labels, text_type, 1, 0.1)
        svm_model(text, labels, text_type, 2, 0.1)
