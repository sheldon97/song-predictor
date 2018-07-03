

# In[1]:
from flask import Flask, render_template

import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix
from flask import Flask, render_template

pd.set_option('display.float_format', lambda x: '%.3f' % x)


user_data = pd.read_csv('/root/Downloads/MillionSongSubset/AdditionalFiles/songs.csv')
user_data.drop_duplicates(subset=['title'])


wide_artist_data = user_data.pivot_table(index = ['title'],columns='artist_id',values = 'artist_hotttnesss').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)
wide_artist_data.head()



from scipy.sparse import csr_matrix

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

save_sparse_csr('/root/last_fm/lastfm_sparse_artist_matrix.npz', wide_artist_data_sparse)



from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_artist_data_sparse)


wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
wide_artist_data_zero_one_sparse = csr_matrix(wide_artist_data_zero_one.values)

save_sparse_csr('/root/last_fm//lastfm_sparse_artist_matrix_binary.npz', wide_artist_data_zero_one_sparse)


model_nn_binary = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn_binary.fit(wide_artist_data_zero_one_sparse)




import string
from fuzzywuzzy import fuzz


def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):

    query_index = None
    ratio_tuples = []


    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))

    return 'Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples])

    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2]
    except:
        return 'Your artist didn\'t match any artists in the data. Try again'

    distances, indices = knn_model.kneighbors(artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors = k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            return 'Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index])
        else:
            print ('{0}: {1}, with distance of {2}:'.format(i, artist_plays_matrix.index[indices.flatten()[i]], distances.flatten()[i]))



print_artist_recommendations('Deep Into The Day,', wide_artist_data_zero_one, model_nn_binary, k = 10)

