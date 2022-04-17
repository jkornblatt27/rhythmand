import re

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob, Word


def ekman_labels(df, label, classifications: list):
    """Creates the Ekman label based on mapping"""
    e_name = 'ekman_' + label
    sub = df[classifications]
    df[e_name] = sub.sum(axis=1)
    df[e_name] = df[e_name].apply(lambda x: 1 if x > 0 else 0)  # Reduce all positive values to 1


def return_emotions(row):
    for col in emotion_sub.columns:
        if row[col] == 1:
            return labels[col]


def clean_text(text):
    """Convert to lower case and remove punctuation"""
    lowered = text.lower()  # Convert text to lowercase
    clean = ' '.join(re.findall(r'([a-zA-Z]+)', lowered))  # keeps letters only
    return clean


def remove_stops(text):
    """Remove stopwords using NLTK"""
    no_stops = ' '.join([word for word in text.split(' ') if word not in stops])  # Remove stopwords
    return no_stops


def textblob_lemmatize(sentence):
    """Get part of speech and then lemmatize word"""
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)


if __name__ == '__main__':

    csvs = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]
    dfs = list()

    # Combine three separate csvs into one
    for csv in csvs:
        df = pd.read_csv(csv, encoding='utf-8')
        dfs.append(df)

    data = pd.concat(dfs)

    # Ekman emotions
    ekman = {
        "anger": ["anger", "annoyance", "disapproval"],
        "disgust": ["disgust"],
        "fear": ["fear", "nervousness"],
        "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                "admiration", "desire", "caring"],
        "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
        "surprise": ["surprise", "realization", "confusion", "curiosity"]
    }

    non_ekman = list()

    # Create Ekman mapping and group all non-Ekman emotions into one list
    for emotion, emotion_list in ekman.items():
        ekman_labels(data, emotion, emotion_list)
        non_ekman = non_ekman + emotion_list

    # Remove all non-Ekman labels
    data = data.drop(non_ekman, axis=1)

    # Create dictionary of Ekman labels and add neutral
    labels = dict()
    for emotion in ekman.keys():
        ekman_label = 'ekman_' + emotion
        labels[ekman_label] = emotion
    labels['neutral'] = 'neutral'

    emotions = list(labels.keys())

    # Remove entries labeled more than one emotion by the rater

    data['total_emotions'] = data[emotions].sum(axis=1)
    data = data.loc[data['total_emotions'] <= 1]
    data = data.drop('total_emotions', axis=1)

    # Create one column for label categories and drop dummies
    emotion_sub = data[emotions]
    data['label'] = emotion_sub.apply(return_emotions, axis=1)
    data = data.drop(emotions, axis=1)

    # Drop null labels (when the example is unclear)
    data = data.loc[data['label'].notnull()]

    # Load stopwords
    stops = stopwords.words('english')

    # Clean the text; apply combination of cleaning techniques to test which works best
    data['text'] = data['text'].apply(clean_text)
    data['clean_no_stops'] = data['text'].apply(remove_stops)
    data['clean_lemmatized'] = data['text'].apply(textblob_lemmatize)
    data['all_processing'] = data['clean_no_stops'].apply(textblob_lemmatize)

    # Drop null values
    data = data.loc[data['all_processing'].notna()]
    data = data.loc[data['all_processing'] != ""]

    # Remove fear & disgust (too few samples), and neutral (emotions only)
    # Get random sample of rest to get even class balance
    data = data.loc[~data['label'].isin(['fear', 'disgust', 'neutral'])]
    n_samples = len(data.loc[data['label'] == 'sadness'])  # Get number of class with fewest samples
    samps = list()
    for emotion in data['label'].unique():
        samp = data.loc[data['label'] == emotion].sample(n=n_samples)
        samps.append(samp)

    data = pd.concat(samps)

    data.to_csv("emotions_clean.csv", index=False, encoding='utf-8')
