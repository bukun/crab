# -*- coding: utf-8 -*-
"""
Base IO code for all datasets
加载数据，将所有的原始数据，
转换成使用以1开始的整数表示的索引
"""

# Authors: Marcel Caraciolo <marcel@muricoca.com>
#          Bruno Melo <bruno@muricoca.com>
# License: BSD Style.

from os.path import dirname
from os.path import join
import numpy as np


class Bunch(dict):
    """
    Container object for datasets: dictionary-like object
    that exposes its keys and attributes. """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def load_movielens_r100k(load_timestamp=False):
    """ Load and return the MovieLens dataset with
        100,000 ratings (only the user ids, item ids, timestamps
        and ratings).

    Parameters
    ----------
    load_timestamp: bool, optional (default=False)
        Whether it loads the timestamp.

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    Examples
    --------
    To load the MovieLens data::

    >>> from scikits.crab.datasets import load_movielens_r100k
    >>> movies = load_movielens_r100k()
    >>> len(movies['data'])
    943
    >>> len(movies['item_ids'])
    1682

    """
    base_dir = join(dirname(__file__), 'data/')
    # Read data
    if load_timestamp:
        data_m = np.loadtxt(base_dir + 'movielens100k.data',
                            delimiter='\t', dtype=int)
        data_movies = {}
        for user_id, item_id, rating, timestamp in data_m:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = (timestamp, int(rating))
    else:
        data_m = np.loadtxt(base_dir + 'movielens100k.data',
                            delimiter='\t', usecols=(0, 1, 2), dtype=int)

        data_movies = {}
        for user_id, item_id, rating in data_m:
            data_movies.setdefault(user_id, {})
            data_movies[user_id][item_id] = int(rating)

    # Read the titles
    data_titles = np.loadtxt(base_dir + 'movielens100k.item',
                             # delimiter='|', usecols=(0, 1), dtype=str)
                             delimiter='|', usecols=(0, 1), dtype=[('f0', int), ('f1', '|S18')])

    data_titles = {item_id: label for (item_id, label) in data_titles}

    fdescr = open(dirname(__file__) + '/descr/movielens100k.rst')

    return Bunch(data=data_movies,
                 item_ids=data_titles,
                 user_ids=None,
                 DESCR=fdescr.read())


def load_sample_songs():
    """ Load and return the songs dataset with
         49 ratings (only the user ids, item ids and ratings).

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    Examples
    --------
    To load the sample songs data::

    >>> from scikits.crab.datasets import load_sample_songs
    >>> songs = load_sample_songs()
    >>> len(songs['data'])
    8
    >>> len(songs['item_ids'])
    8

    """
    base_dir = join(dirname(__file__), 'data/')

    # Read data
    data_m = np.loadtxt(base_dir + 'sample_songs.csv',
                         delimiter=',',
                        dtype=[('f0', '|S18'), ('f1', '|S18'), ('f2', float)])

    # data_m = np.loadtxt(base_dir + 'sample_movies.csv', delimiter=';',
    #                     dtype=[('f0', '|S18'), ('f1', '|S18'), ('f2', float)])
    #                     # dtype=[('f0', str), ('f1', str), ('f2', float)])

    item_ids = []
    user_ids = []
    data_songs = {}
    for user_id, item_id, rating in data_m:
        if user_id not in user_ids:
            user_ids.append(user_id)
        if item_id not in item_ids:
            item_ids.append(item_id)
        u_ix = user_ids.index(user_id) + 1
        i_ix = item_ids.index(item_id) + 1
        data_songs.setdefault(u_ix, {})
        # data_songs[u_ix][i_ix] = float(rating)
        data_songs[u_ix][i_ix] = rating


    fdescr = open(dirname(__file__) + '/descr/sample_songs.rst')

    return Bunch(data=data_songs,
                 item_ids={no + 1: item_id for no, item_id in enumerate(item_ids)},
                 user_ids={no + 1: user_id for no, user_id in enumerate(user_ids)},
                 DESCR=fdescr.read())


def load_sample_movies():
    """ Load and return the movies dataset with
         n ratings (only the user ids, item ids and ratings).

    Return
    ------
    data: Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the full data in the shape:
            {user_id: { item_id: (rating, timestamp),
                       item_id2: (rating2, timestamp2) }, ...} and
        'user_ids': the user labels with respective ids in the shape:
            {user_id: label, user_id2: label2, ...} and
        'item_ids': the item labels with respective ids in the shape:
            {item_id: label, item_id2: label2, ...} and
        DESCR, the full description of the dataset.

    Examples
    --------
    To load the sample movies data::

    >>> from scikits.crab.datasets import load_sample_movies
    >>> movies = load_sample_movies()
    >>> len(movies['data'])
    7
    >>> len(movies['item_ids'])
    6

    """
    base_dir = join(dirname(__file__), 'data/')

    # Read data
    # data_m = np.loadtxt(base_dir + 'sample_movies.csv',                delimiter=';', dtype=str)

    data_m = np.loadtxt(base_dir + 'sample_movies.csv', delimiter=';',
                        dtype=[('f0', '|S18'), ('f1', '|S18'), ('f2', float)])
                        # dtype=[('f0', str), ('f1', str), ('f2', float)])

    item_ids = []
    user_ids = []
    data_songs = {}
    for user_id, item_id, rating in data_m:
        if user_id not in user_ids:
            user_ids.append(user_id)
        if item_id not in item_ids:
            item_ids.append(item_id)
        u_ix = user_ids.index(user_id) + 1
        i_ix = item_ids.index(item_id) + 1
        # Python 字典(Dictionary) setdefault() 函数和get()方法类似, 如果键不已经存在于字典中，将会添加键并将值设为默认值。
        # Python 字典(Dictionary) get() 函数返回指定键的值，如果值不在字典中返回默认值。
        data_songs.setdefault(u_ix, {})
        # data_songs[u_ix][i_ix] = float(rating.astype(np.str))
        data_songs[u_ix][i_ix] = rating


    fdescr = open(dirname(__file__) + '/descr/sample_movies.rst')

    return Bunch(data=data_songs,
                 item_ids={ no +1: item_id for no, item_id in enumerate(item_ids)},
                 user_ids={ no + 1: user_id for no, user_id in enumerate(user_ids)},
                 DESCR=fdescr.read())

if __name__ == '__main__':
    load_movielens_r100k()