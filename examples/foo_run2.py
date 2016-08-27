#!/usr/bin/env python
# coding=utf-8

def base_demo():
    # 基础数据-测试数据
    from scikits.crab import datasets
    movies = datasets.load_sample_movies()
    # print movies.data
    # print movies.user_ids
    # print movies.item_ids

    # Build the model
    from scikits.crab.models import MatrixPreferenceDataModel
    model = MatrixPreferenceDataModel(movies.data)

    # Build the similarity
    # 选用算法 pearson_correlation
    from scikits.crab.metrics import pearson_correlation
    from scikits.crab.similarities import UserSimilarity
    similarity = UserSimilarity(model, pearson_correlation)

    # 选择 基于User的推荐
    from scikits.crab.recommenders.knn import UserBasedRecommender
    recommender = UserBasedRecommender(model, similarity, with_preference=True)
    print (recommender.recommend(5))  # 输出个结果看看效果 Recommend items for the user 5 (Toby)

    # 选择 基于Item 的推荐(同样的基础数据，选择角度不同)
    from scikits.crab.recommenders.knn import ItemBasedRecommender
    recommender = ItemBasedRecommender(model, similarity, with_preference=True)
    print (recommender.recommend(5))  # 输出个结果看看效果 Recommend items for the user 5 (Toby)


def itembase_demo():
    from scikits.crab.models.classes import MatrixPreferenceDataModel
    from scikits.crab.recommenders.knn.classes import ItemBasedRecommender
    from scikits.crab.similarities.basic_similarities import ItemSimilarity
    from scikits.crab.recommenders.knn.item_strategies import ItemsNeighborhoodStrategy
    from scikits.crab.metrics.pairwise import euclidean_distances
    movies = {
        'Marcel Caraciolo': \
            {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5,
             'You, Me and Dupree': 2.5, 'The Night Listener': 3.0}, \
        'Paola Pow': \
            {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0,
             'The Night Listener': 3.0, 'You, Me and Dupree': 3.5}, \
        'Leopoldo Pires': \
            {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5, 'The Night Listener': 4.0},
        'Lorena Abreu': \
            {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0,
             'You, Me and Dupree': 2.5}, \
        'Steve Gates': \
            {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,
             'The Night Listener': 3.0, 'You, Me and Dupree': 2.0}, \
        'Sheldom': \
            {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0, 'Superman Returns': 5.0,
             'You, Me and Dupree': 3.5}, \
        'Penny Frewman': \
            {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}, 'Maria Gabriela': {}
    }
    model = MatrixPreferenceDataModel(movies)
    items_strategy = ItemsNeighborhoodStrategy()
    similarity = ItemSimilarity(model, euclidean_distances)
    recsys = ItemBasedRecommender(model, similarity, items_strategy)

    print( recsys.most_similar_items('Lady in the Water'))
    # Return the recommendations for the given user.
    print (recsys.recommend('Leopoldo Pires'))
    # Return the 2 explanations for the given recommendation.
    print (recsys.recommended_because('Leopoldo Pires', 'Just My Luck', 2))
    # Return the similar recommends
    print (recsys.most_similar_items('Lady in the Water'))
    # 估算评分
    print (recsys.estimate_preference('Leopoldo Pires', 'Lady in the Water'))


base_demo()
itembase_demo()

