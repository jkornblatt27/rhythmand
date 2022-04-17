import os
from string import capwords

from flask import Flask, request, render_template
import psycopg2

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/results', methods=['POST'])
def results():
    # Connect to the database
    DATABASE_URL = os.environ.get('HEROKU_POSTGRESQL_COPPER_URL')
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Get artist and album names from form
    artist = request.form.get("artist").strip().lower()
    album = request.form.get("album").strip().lower()

    query = """SELECT * from albums where lower(artist) = %(artist)s and lower(album) = %(album)s"""
    cur.execute(query, {'artist': artist, 'album': album})
    results = cur.fetchall()

    if len(results) == 0:
        error_message1 = 'The album {album} by {artist} was not found in the database.'.format(album=capwords(album),
                                                                                               artist=capwords(artist))
        error_message2 = 'Please try a different album, or ensure the album and artist names are spelled correctly ' \
                         'and try again.'
        return render_template('home.html', output_1=error_message1, output_2=error_message2)
    else:

        album_art = results[0][4]
        album_mood = results[0][5]

        # Have the results page use a different color for each mood
        mood_colors = {
            'anger': '#DC143C',
            'joy': '#FFD700',
            'sadness': '#00008B',
            'surprise': '#8B008B'
        }

        mood_color = mood_colors[album_mood]

        # Get indices of five most similar albums
        id1 = results[0][6]
        id2 = results[0][7]
        id3 = results[0][8]
        id4 = results[0][9]
        id5 = results[0][10]

        top5query = """SELECT artist, album, album_art from albums where id = %(id1)s
        UNION ALL
        SELECT artist, album, album_art from albums where id = %(id2)s
        UNION ALL
        SELECT artist, album, album_art from albums where id = %(id3)s
        UNION ALL 
        SELECT artist, album, album_art from albums where id = %(id4)s 
        UNION ALL
        SELECT artist, album, album_art from albums where id = %(id5)s
        """
        cur.execute(top5query, {'id1': id1, 'id2': id2, 'id3': id3, 'id4': id4, 'id5': id5})
        top5 = cur.fetchall()
        top_five_artist = [result[0] for result in top5]
        top_five_album = [result[1] for result in top5]
        top_five_art = [result[2] for result in top5]
        cur.close()
        conn.close()
        return render_template('results.html', album=capwords(album), artist=capwords(artist), album_mood=album_mood,
                               album_art=album_art, mood_color=mood_color, top_five_artist=top_five_artist, top_five_album=top_five_album,
                               top_five_art=top_five_art)


if __name__ == '__main__':
    app.run(debug=True)
