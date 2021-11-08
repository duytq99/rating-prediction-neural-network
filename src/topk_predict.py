import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
from prettytable import PrettyTable
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from metrics import RMSE
from get_data import *
from model import *

with tf.device("cpu:0"):
   pass

# load prediction model
model = load_model('model_checkpoint/best_project_cosine_warmup.hdf5')

# load data
movies_df = pd.read_csv('ml-20m/movies.csv')
movie_genome_df = pd.read_csv('data/genome_scores_filtered_20212010_namdn.csv')
movie_genome_array = movie_genome_df.relevance.values.reshape(-1,1128)
movie_group = movie_genome_df.groupby(movie_genome_df.movieId).mean()

# load train/test data
df_total = pd.read_csv('data/filtered_ratings_20212010_namdn.csv')
df_train = pd.read_csv('data/training_set_70_20212010_namdn.csv')
df_test = pd.read_csv('data/testing_set_10_20212010_namdn.csv')
movie_id_all = df_total.movieId.unique()

# input user ID
user_list = df_train.userId.unique()
print('Chose user ID from list: ', user_list[:100])
user_id = int(input('Enter user id: '))
while user_id not in user_list:
    print('Invalid!')
    user_id = int(input('Re-enter user id: '))
movie_id_train = df_train[df_train['userId']==user_id].movieId.values
movie_id_test = df_test[df_test['userId']==user_id].movieId.values
user_genome = np.around(get_user_genome(user_id, df_train), 5)


# define KNN model
K = int(input("Enter K: "))
neigh = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=2*K, n_jobs=-1)
neigh.fit(movie_genome_array)

# find K movies using
distances, top_K_id = neigh.kneighbors(user_genome, return_distance=True)
movie_ids = movie_group.iloc[top_K_id[0]].index.values


# get prediction
movie_genomes = movie_genome_array[top_K_id].squeeze()
user_genome = np.repeat(user_genome, movie_genomes.shape[0], axis=0)
predictions = model.predict([user_genome, movie_genomes])

# sort predictions
# print(predictions)
# print(movie_ids)
sorted_index = predictions.argsort(axis=0)
predictions = predictions[sorted_index[::-1]].squeeze()
movie_ids = movie_ids[sorted_index.reshape(1,-1)][0].squeeze()
# print(predictions)
# print(movie_ids)
# quit()

# print movie names
names, genres = get_movie_name(movie_ids, movies_df)

log = PrettyTable()
log.field_names = ["Movie name", "Genre", "Distance", "Estimated rating", "Ground truth"]
for i in range(len(names)//2):
    gt_rating = get_groundtruth_rating(user_id, movie_ids[i], df_total, df_total, movie_id_train, movie_id_test)
    log.add_row(
        [names[i][0], 
        genres[i][0], 
        np.around(distances[:,i][0],3), 
        np.around(predictions[i], 3), 
        gt_rating[0]]
        )

print(log)
