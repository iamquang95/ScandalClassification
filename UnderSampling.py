#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy
import codecs
import random
from math import sqrt 


from Queue import PriorityQueue

from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

import TextProcessUtils

def undersampling(train_dict, train_label, random_seed):
    # static seed
    print("[INFO] START undersamping")
    random.seed(random_seed)
    pos_index = []
    neg_index = []
    for (i, label) in enumerate(train_label):
        if (label == 1):
            pos_index.append(i)
        else:
            neg_index.append(i) 
    nNeg = len(neg_index)
    nPos = len(pos_index)

    selected_neg = random.sample(range(0, nNeg), nPos)

    new_train_dict = []
    new_train_label = []

    for i in pos_index:
        new_train_dict.append(train_dict[i])
        new_train_label.append(train_label[i])
    for i in selected_neg:
        new_train_dict.append(train_dict[neg_index[i]])
        new_train_label.append(train_label[neg_index[i]])

    print("[INFO] FINISH undersamping, remove %s negative samples" % (nNeg - nPos))


    return (new_train_dict, new_train_label)




