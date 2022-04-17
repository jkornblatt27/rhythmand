import json
import socket
import logging
import pickle

import pandas as pd
import numpy as np
import yaml
import requests
from requests.exceptions import RequestException
from lyricsgenius import Genius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from cleaning import clean_text, textblob_lemmatize

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

logging.basicConfig(filename='create_dataset.log', level=logging.ERROR)


def score_lyrics(lyrics):
    """Returns a dictionary of emotion scores for each song"""
    lyrics = countvectorizer.transform(lyrics)
    lyrics = tfidfvectorizer.transform(lyrics)
    probs = model._predict_proba_lr(lyrics)[0]
    classes = model.classes_
    score_dict = {}
    for i in range(len(probs)):
        score_dict[classes[i]] = probs[i]
    return score_dict


def emotion_averages(score_dict):
    """Returns average emotion scores for a collection of songs"""
    em_avgs = {}
    emotions = ['anger', 'joy', 'sadness', 'surprise']
    for emotion in emotions:
        song_scores = list()
        for score in score_dict:
            song_scores.append(score[emotion])
        em_avgs[emotion] = np.mean(song_scores)
    return em_avgs


def genius_score_album(album, artist):
    """Return the emotion scores for the album as a whole"""
    emotions = ['anger', 'joy', 'sadness', 'surprise']
    try:
        album = genius.search_album(album, artist)
    # except requests.exceptions.RequestException as e:
    except requests.exceptions.Timeout as e:
        print(e)
        logging.error('Timeout on album with index {}'.format(str(i)))
        genius_score = {key: None for key in emotions}
    else:
        song_stop_words = ['verse', 'chorus', 'bridge', 'lyric', 'outro', 'intro', 'embed']
        score_dicts = list()
        if album is not None:
            tracks = album.to_dict()['tracks']
            for track in tracks:
                lyrics = track['song']['lyrics']
                if lyrics != '':
                    lyrics = lyrics.split('Lyrics')[1]  # Lyrics in the song always come after "<Title> Lyrics"
                    lyrics = textblob_lemmatize(clean_text(lyrics))
                    for word in song_stop_words:
                        lyrics = lyrics.replace(word, '')
                    score_dicts.append(score_lyrics([lyrics]))
            genius_score = emotion_averages(score_dicts)
        else:
            genius_score = {key: None for key in emotions}
    return genius_score


def get_spotify_data(artist, album):
    """Collect sonic attributes and artwork for an album"""
    feats_list = list()
    try:
        album = sp.search(q='artist:' + artist + ' album:' + album)['tracks']['items'][0]['album']['id']
        art_link = sp.album(album)['images'][0]['url']
        tracks = sp.album(album)['tracks']['items']
        for track in tracks:
            track_id = track['id']
            feats = sp.audio_features(track_id)
            feats_list.append(feats)
    except:
        feats_list = None
        art_link = None
    return feats_list, art_link


def create_spotify_dict(artist, album):
    """Return the Spotify sonic features for an album as a whole"""
    spotify_attributes = {}
    spotify_attributes['artist'] = artist
    spotify_attributes['album'] = album
    feats, album_art = get_spotify_data(artist, album)
    feature_categories = ['danceability', 'energy', 'acousticness', 'loudness']
    if feats is not None:
        for feature in feature_categories:
            scores = list()
            for feat in feats:
                if feat:
                    if feat[0]:
                        scores.append(feat[0][feature])
                else:
                    scores.append(np.nan)
            spotify_attributes[feature] = np.nanmean(scores)
        spotify_attributes['album_art'] = album_art
    else:
        feature_categories.append(album_art)
        feats_dict = {key: None for key in feature_categories}
        spotify_attributes = {**spotify_attributes, **feats_dict}
    return spotify_attributes


if __name__ == '__main__':
    # Connect to Genius and Spotify APIs
    genius = Genius(config['Genius']['Client_Access_Token'], verbose=False)
    client_credentials_manager = SpotifyClientCredentials(config['Spotify']['Client_ID'],
                                                          config['Spotify']['Client_Secret'])
    sp = spotipy.Spotify(auth_manager=client_credentials_manager)

    # Load model and vectorizers
    model = pickle.load(open("clean_lemmatizedsvm_2_gram_0.1_c_model.pickle", 'rb'))
    countvectorizer = pickle.load(open("clean_lemmatizedsvm_2_gram_0.1_c_countvectorizer.pickle", 'rb'))
    tfidfvectorizer = pickle.load(open("clean_lemmatizedsvm_2_gram_0.1_c_tfidfvectorizer.pickle", 'rb'))

    # Load in pitchfork albums
    data = pd.read_csv("pitchfork.csv")

    sub = data
    albums = sub['album'].tolist()
    artists = sub['artist'].tolist()
    scores = list()

    for i in range(len(data)):
        album = albums[i]
        artist = artists[i]
        sd = create_spotify_dict(artist, album)
        gd = genius_score_album(album, artist)
        combined_scores = {**sd, **gd}
        scores.append(combined_scores)

        if (i + 1) // 10 > i // 10:
            scores_json = json.dumps(scores)
            sub_num = str((i // 10) + 1)
            # pathname = 'albums/albums' + sub_num + '.json'
            pathname = 'albums_test/albums' + sub_num + '.json'
            with open(pathname, 'w') as outfile:
                json.dump(scores_json, outfile)
            scores = list()
