import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import keras
import pandas as pd
import numpy as np

from metrics import RMSE
from get_data import *
from model import *

with tf.device("cpu:0"):
   pass

# load prediction model
model = load_model('model_checkpoint/best_project_ver2.hdf5')

# load train/test data
df_total = pd.read_csv('data/filtered_ratings_20212010_namdn.csv')
df_train = pd.read_csv('data/training_set_70_20212010_namdn.csv')
df_test = pd.read_csv('data/testing_set_10_20212010_namdn.csv')

# input user id
user_list = df_total.userId.unique()
print('Chose user ID from list: ', user_list[:100])
user_id = int(input('Enter user id: '))
while user_id not in user_list:
    print('Invalid!')
    user_id = int(input('Re-enter user id: '))

# id of movie in train set/test set
movie_id_all = df_total.movieId.unique()
movie_id_train = df_train[df_train['userId']==user_id].movieId.values
movie_id_test = df_test[df_test['userId']==user_id].movieId.values
print('Movie id in train set', movie_id_train)
print('Movie id in test set', movie_id_test)

# input movie id -> predict rating
movie_id = int(input('Enter movie id: '))
while movie_id not in movie_id_all:
    print('Invalid!')
    movie_id = int(input('Re-enter movie id: '))
gt_rating = get_groundtruth_rating(user_id, movie_id, df_train, df_test, movie_id_train, movie_id_test)
names, genres = get_movie_name([movie_id], pd.read_csv('ml-20m/movies.csv'))

# Print movie infor
print('-----------------------------------------')
print('Movie name: ', names[0][0])
print('Movie genre: ', genres[0][0])
if gt_rating is not None:
    print('Actual rating: ', gt_rating[0])
else: 
    print('No rating')

# inference -> get prediction
movie_genome = get_movie_genome(movie_id)
print('Movie genome', movie_genome)
user_genome = np.around(get_user_genome(user_id, df_train), 5)
print('User genome', user_genome)
predicted_rating = model.predict([user_genome, movie_genome])
print('-----------------------------------------\nPredicted rating: ', predicted_rating[0,0])
print('Done!')