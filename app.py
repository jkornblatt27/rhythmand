import pickle
import re
from string import capwords
import os

import requests
from flask import Flask, request, render_template
import pandas as pd
import sklearn
from lyricsgenius import Genius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from app_helpers import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results', methods=['POST'])
def results():
    # Load data
    data = pd.read_csv("album_db.csv")
    # Connect to Genius and Spotify APIs
    genius = Genius(os.environ.get('GENIUS_CLIENT_ACCESS_TOKEN'), verbose=False)
    client_credentials_manager = SpotifyClientCredentials(os.environ.get('SPOTIFY_CLIENT_ID'),
                                                          os.environ.get('SPOTIFY_CLIENT_SECRET'))
    sp = spotipy.Spotify(auth_manager=client_credentials_manager)

    # Load model and vectorizers
    model = pickle.load(open("clean_lemmatizedsvm_2_gram_0.1_c_model.pickle", 'rb'))
    countvectorizer = pickle.load(open("clean_lemmatizedsvm_2_gram_0.1_c_countvectorizer.pickle", 'rb'))
    tfidfvectorizer = pickle.load(open("clean_lemmatizedsvm_2_gram_0.1_c_tfidfvectorizer.pickle", 'rb'))

    # Get artist and album names from form
    artist = request.form.get("artist").strip().lower()
    album = request.form.get("album").strip().lower()

    try:
        genius_album, spotify_album = validate_album(genius, sp, album, artist)
    except requests.exceptions.Timeout:
        # Return an error message if the operation times out
        error_message = 'The operation timed out. Please ensure the artist and album names are ' \
                        'spelled correctly and try again.'
        return render_template('home.html', output=error_message)
    except Exception:
        # Return an error message if the album is not found
        error_message = '{album} by {artist} not found. ' \
                        'Please ensure the artist and album names ' \
                        'are spelled correctly and try again.'.format(album=capwords(album), artist=capwords(artist))
        return render_template('home.html', output=error_message)
        # Gather Genius and Spotify data on the album
    else:
        genius_album_score, genius_song_scores = genius_score_album(genius_album, countvectorizer,
                                                                    tfidfvectorizer,
                                                                    model)  # Get raw emotion scores using model
        album_mood, song_moods = get_dominant_emotions(genius_album_score,
                                                       genius_song_scores)  # Get dominant mood for album/songs

        # Have the results page use a different color for each mood
        mood_colors = {
            'anger': '#DC143C',
            'joy': '#FFD700',
            'sadness': '#00008B',
            'surprise': '#8B008B'
        }

        mood_color = mood_colors[album_mood]
        song_mood_colors = [mood_colors[list(i.values())[0]] for i in song_moods]
        n_songs = len(song_moods)

        feats, album_art = get_spotify_data(sp, spotify_album)  # Spotify data
        spot_dict = create_spotify_dict(feats)
        all_feats = {**spot_dict, **genius_album_score}  # Combine Spotify and scored Genius data into one dictionary
        data = data.loc[data['artist'].apply(
            lambda x: x.lower()) != artist]  # Ensure no albums from the same artist are included in recommendation
        data = data.append(all_feats, ignore_index=True)
        data['id'] = data.index
        album_info = data[["id", "artist", "album", "album_art"]]
        album_features = data[
            ["id", "danceability", "energy", "acousticness", "loudness", "anger", "joy", "sadness", "surprise"]]
        album_features = album_features.set_index("id")
        most_similar = get_similarity_indices(album_features)
        sim = pd.DataFrame(most_similar, columns=['id'])
        top_five = pd.merge(sim, album_info, on='id')
        top_five_artist = top_five['artist'].tolist()
        top_five_album = top_five['album'].tolist()
        top_five_art = top_five['album_art'].tolist()
        return render_template('results.html', album=capwords(album), artist=capwords(artist), album_mood=album_mood,
                               song_moods=song_moods, album_art=album_art, mood_color=mood_color,
                               song_mood_colors=song_mood_colors, n_songs=n_songs, top_five_artist=top_five_artist,
                               top_five_album=top_five_album, top_five_art=top_five_art)


if __name__ == '__main__':
    app.run(debug=True)
