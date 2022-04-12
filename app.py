from string import capwords

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

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

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results', methods=['POST'])
def results():
    # Load data
    data = pd.read_csv("album_db.csv")

    # Get artist and album names from form
    artist = request.form.get("artist").strip().lower()
    album = request.form.get("album").strip().lower()

    requested_album = data.loc[
        (data['artist'].apply(lambda x: x.lower()) == artist) & (data['album'].apply(lambda x: x.lower()) == album)]

    if len(requested_album) == 0:
        error_message1 = 'The album {album} by {artist} was not found in the database.'.format(album=capwords(album),
                                                                                               artist=capwords(artist))
        error_message2 = 'Please try a different album, or ensure the album and artist names are spelled correctly ' \
                         'and try again.'
        return render_template('home.html', output_1=error_message1, output_2=error_message2)
    else:
        album_art = requested_album['album_art'].iloc[0]
        emotions = requested_album[["anger", "joy", "sadness", "surprise"]].reset_index(drop=True)
        album_mood = emotions.idxmax(axis=1)[0]

        # Have the results page use a different color for each mood
        mood_colors = {
            'anger': '#DC143C',
            'joy': '#FFD700',
            'sadness': '#00008B',
            'surprise': '#8B008B'
        }

        mood_color = mood_colors[album_mood]

        data = data.loc[data['artist'].apply(
            lambda x: x.lower()) != artist]  # Ensure no albums from the same artist are included in recommendation
        data = data.append(requested_album, ignore_index=True)
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
                               album_art=album_art, mood_color=mood_color, top_five_artist=top_five_artist,
                               top_five_album=top_five_album, top_five_art=top_five_art)


if __name__ == '__main__':
    app.run(debug=True)
