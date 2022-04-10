import re

from textblob import TextBlob, Word
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def clean_text(text):
    """Convert to lower case and remove punctuation"""
    lowered = text.lower()  # Convert text to lowercase
    clean = ' '.join(re.findall(r'([a-zA-Z]+)', lowered))  # keeps letters only
    return clean


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


def validate_album(genius, sp, album, artist):
    """Retrieve album data from Genius and Spotify"""
    genius_album = genius.search_album(album, artist)
    spotify_album = sp.search(q='artist:' + artist + ' album:' + album)['tracks']['items'][0]['album']['id']
    spotify_album = sp.album(spotify_album)
    return genius_album, spotify_album


def score_lyrics(countvectorizer, tfidfvectorizer, model, lyrics):
    """Use the emotion to get emotion probabilities for each song"""
    lyrics = countvectorizer.transform(lyrics)
    lyrics = tfidfvectorizer.transform(lyrics)
    probs = model._predict_proba_lr(lyrics)[0]
    classes = model.classes_
    score_dict = {}
    for i in range(len(probs)):
        score_dict[classes[i]] = probs[i]
    return score_dict


def emotion_averages(score_dict):
    """For an album, get the averages for the emotion of each song"""
    em_avgs = {}
    emotions = ['anger', 'joy', 'sadness', 'surprise']
    for emotion in emotions:
        song_scores = list()
        for score in score_dict:
            song_scores.append(score[emotion])
        em_avgs[emotion] = np.mean(song_scores)
    return em_avgs


def genius_score_album(album, countvectorizer, tfidfvectorizer, model):
    """Get the emotion probabilities for the album as a whole and for individual songs"""
    song_stop_words = ['verse', 'chorus', 'bridge', 'lyric', 'outro', 'intro', 'embed']
    song_scores = list()
    score_dicts = list()
    tracks = album.to_dict()['tracks']
    for track in tracks:
        lyrics = track['song']['lyrics']
        if lyrics != '':
            lyrics = lyrics.split('Lyrics')[1]  # Lyrics in the song always come after "<Title> Lyrics"
            lyrics = textblob_lemmatize(clean_text(lyrics))
            for word in song_stop_words:
                lyrics = lyrics.replace(word, '')
            scores = score_lyrics(countvectorizer, tfidfvectorizer, model, [lyrics])
            song_dict = dict()
            song_dict[track['song']['title']] = scores
            song_scores.append(song_dict)
            score_dicts.append(scores)
    album_score = emotion_averages(score_dicts)
    return album_score, song_scores


def get_dominant_emotions(album_score, song_scores):
    """Get the dominant emotion for the album and for each song"""
    album_mood = max(album_score, key=album_score.get)
    song_moods = list()
    for song in song_scores:
        mood_dict = dict()
        mood_dict[list(song.keys())[0]] = max(list(song.values())[0], key=(list(song.values())[0]).get)
        song_moods.append(mood_dict)
    return album_mood, song_moods


def get_spotify_data(sp, sp_album):
    """Get audio features and album art for requested album"""
    feats_list = list()
    art_link = sp_album['images'][0]['url']
    tracks = sp_album['tracks']['items']
    for track in tracks:
        track_id = track['id']
        feats = sp.audio_features(track_id)
        feats_list.append(feats)
    return feats_list, art_link


def create_spotify_dict(feats_list):
    """Get averages of important features for the album as a whole"""
    spotify_attributes = {}
    feature_categories = ['danceability', 'energy', 'acousticness', 'loudness']
    for feature in feature_categories:
        scores = list()
        for feat in feats_list:
            if feat:
                if feat[0]:
                    scores.append(feat[0][feature])
            else:
                scores.append(np.nan)
        spotify_attributes[feature] = np.nanmean(scores)
    return spotify_attributes


def get_similarity_indices(data):
    """Get the indices of the five most similar albums using cosine similarity"""
    scaler = MinMaxScaler()
    for col in data.columns:
        data[col] = scaler.fit_transform(
            np.array(data[col]).reshape(-1, 1))  # Normalize  each column so they all scale from (0, 1)
    cos_sim = cosine_similarity(data, data)  # Create cosine similarity matrix
    sim_scores = list(enumerate(cos_sim[len(cos_sim) - 1]))  # Get cosine similarity scores for requested album
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    inds = [i[0] for i in sim_scores[1:6]]  # Get the indices of the top five most similar albums
    return inds
