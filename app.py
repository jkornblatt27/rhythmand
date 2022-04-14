import os
from string import capwords

from flask import Flask, request, render_template
import psycopg2
import pandas as pd
import numpy as np


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results', methods=['POST'])
def results():

    # Connect to the database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Get artist and album names from form
    artist = request.form.get("artist").strip().lower()
    album = request.form.get("album").strip().lower()

    query = """SELECT * from albums where lower(artist) = %(artist)s and lower(album) = %(album)s"""
    cur.execute(query, {'artist': artist, 'album': album})
    results = cur.fetchall()

    if len(requested_album) == 0:
        error_message1 = 'The album {album} by {artist} was not found in the database.'.format(album=capwords(album),
                                                                                               artist=capwords(artist))
        error_message2 = 'Please try a different album, or ensure the album and artist names are spelled correctly ' \
                         'and try again.'
        return render_template('home.html', output_1=error_message1, output_2=error_message2)
    else:

        # Have the results page use a different color for each mood
        album_mood = results[5]

        mood_colors = {
            'anger': '#DC143C',
            'joy': '#FFD700',
            'sadness': '#00008B',
            'surprise': '#8B008B'
        }

        mood_color = mood_colors[album_mood]

        # Get indices of five most similar albums
        id1 = results[6]
        id2 = results[7]
        id3 = results[8]
        id4 = results[9]
        id5 = results[10]

        top5query = f"""SELECT artist, album from albums where id = {id1} 
        UNION ALL
        SELECT artist, album from albums where id = {id2} 
        UNION ALL
        SELECT artist, album from albums where id = {id3}
        UNION ALL 
        SELECT artist, album from albums where id = {id4} 
        UNION ALL
        SELECT artist, album from albums where id = {id5}
        """
        cur.execute(top5query)
        top5 = cur.fetchall()
        top_five_artist = [result[0] for result in top5]
        top_five_album = [result[1] for result in top5]
        return render_template('results.html', album=capwords(album), artist=capwords(artist), album_mood=album_mood,
                               mood_color=mood_color, top_five_artist=top_five_artist, top_five_album=top_five_album)

        cur.close()
        conn.close()


if __name__ == '__main__':
    app.run(debug=True)
