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

def oversampling(train_dict, train_label, random_seed):
    # static seed
    print("[INFO] START oversamping")
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

    more_pos_samples = (nNeg - nPos)

    duplicate_pos = []
    for i in xrange(more_pos_samples):
        duplicate_pos.append(random.randint(0, nPos-1))

    for index in duplicate_pos:
        train_dict.append(train_dict[pos_index[index]])
        train_label.append(train_label[pos_index[index]])

    print("[INFO] FINISH oversamping, add %s more positive samples" % (more_pos_samples))


    return (train_dict, train_label)




