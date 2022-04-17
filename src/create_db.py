import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    # Read in data and get dominant emotion for each album
    data = pd.read_csv("album_db.csv")
    data = data.reset_index().rename(columns={'index': 'id'})
    d_sub = data[["anger", "joy", "sadness", "surprise"]]
    max_emotions = d_sub.idxmax(axis=1).reset_index().rename(columns={'index': 'id', 0: 'top_emotion'})
    data = pd.merge(data, max_emotions, on='id')

    # Partition data into relevant features for recommendation system
    album_info = data[["id", "artist", "album", "album_art", "top_emotion"]]
    album_features = data[
        ["id", "danceability", "energy", "acousticness", "loudness", "anger", "joy", "sadness", "surprise"]]
    album_features = album_features.set_index("id")

    # Apply MinMaxScaler to represent features as values relative to dataset
    scaler = MinMaxScaler()
    for col in album_features.columns:
        album_features[col] = scaler.fit_transform(np.array(album_features[col]).reshape(-1, 1))

    # Get the five most similar albums for each album
    cos_sim = cosine_similarity(album_features, album_features)
    ids = list()
    top5 = list()
    for i in range(len(cos_sim)):
        sim_scores = list(enumerate(cos_sim[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        ids.append(i)
        inds = [i[0] for i in sim_scores[1:6]]
        top5.append(inds)

    # Create separate columns for each album
    tops = pd.DataFrame(list(zip(ids, top5)), columns=['id', 'top5'])
    for i in range(5):
        colname = 'id' + str(i + 1)
        tops[colname] = tops['top5'].apply(lambda x: x[i])

    tops = tops.drop('top5', axis=1)
    data = pd.merge(album_info, tops, on='id')
    data.to_csv("albums_db_final.csv", index=False)
