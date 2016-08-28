# -*- coding: utf-8 -*-

from scikits.crab import datasets
from scikits.crab.recommenders.knn import UserBasedRecommender

movies = datasets.load_sample_movies()
songs = datasets.load_sample_songs()
print (movies.data)
print(songs.data)

print( movies.user_ids)
print( songs.user_ids)

print(movies.item_ids)
print(songs.item_ids)

from scikits.crab.models import MatrixPreferenceDataModel
model = MatrixPreferenceDataModel(movies.data)

from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
#Build the similarity
similarity = UserSimilarity(model,pearson_correlation)

from scikits.crab.recommenders.knn import UserBasedRecommender

recommender = UserBasedRecommender(model, similarity,with_preference=True)

print(recommender.recommend(5))