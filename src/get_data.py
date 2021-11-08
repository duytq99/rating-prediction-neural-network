import pandas as pd
import numpy as np

movie_genome_df = pd.read_csv('data/genome_scores_filtered_20212010_namdn.csv')

def get_groundtruth_rating(user_id, movie_id, df_train, df_test, movie_id_train, movie_id_test):
    # if movie id in train set, print ground truth and estimated rating
    if movie_id in movie_id_train:
        gt_rating = df_train[(df_train['userId']==user_id) & (df_train['movieId']==movie_id)].rating.values
    # if movie id in test set, print ground truth and estimated rating
    elif movie_id in movie_id_test:
        gt_rating = df_test[(df_test['userId']==user_id) & (df_test['movieId']==movie_id)].rating.values
    # if movie is not in train and test set -> new movie for user, there is not groundtruth
    else:
        gt_rating = [None]
    return gt_rating

def get_user_genome(user_id, df_train):
    max_value = df_train['rating'].max()
    min_value = df_train['rating'].min()
    df_train['rating'] = (df_train['rating'] - min_value) / (max_value - min_value)
    total_feature = 0
    users = df_train[df_train.userId == user_id].values
    # print(users)
    for _, movie_id, rating, _ in users:
        # print(movie_id)
        genome = movie_genome_df.relevance[movie_genome_df.movieId==movie_id].values
        total_feature += genome*rating
    return np.divide(total_feature, len(users)).reshape(1,-1)

def get_movie_genome(movie_id):
    return movie_genome_df[movie_genome_df.movieId==movie_id].relevance.values.reshape(-1,1128)
    

def get_movie_name(movieIds, df):
    name = []
    genres = []
    for id in movieIds:
        name.append(df[df.movieId==id].title.values)
        genres.append(df[df.movieId==id].genres.values)
    return name, genres


if __name__=='__main__':
    # test get_movie_name()
    movies_df = pd.read_csv('ml-20m/movies.csv')
    movie_ids = np.random.randint(1, 10000, 5)
    print(movie_ids)
    names, genres = get_movie_name(movie_ids, movies_df)
    for i in range(len(names)):
        print(names[i], genres[i])