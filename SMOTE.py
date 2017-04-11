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

def scalar(collection): 
  total = 0 
  for coin, count in collection.items(): 
    total += count * count 
  return sqrt(total) 

def similarity(A,B): # A and B are coin collections 
  total = 0 
  for kind in A: # kind of coin 
    if kind in B: 
      total += A[kind] * B[kind] 
  return float(total) / (scalar(A) * scalar(B))

def distance(dict1, dict2):
    similarity(dict1, dict2)

def populateData(dicts, rate, d1, neighborsPQ):
    newSamples = []
    neighbors = []
    while (not neighborsPQ.empty()):
        (dist, j) = neighborsPQ.get()
        neighbors.append(j)
    for i in xrange(0, rate):
        nn = random.randint(0, len(neighbors) - 1)
        newSample = {}
        d2 = dicts[neighbors[nn]]
        for key in d1.keys():
            dif = d2.get(key, 0) - d1[key]
            gap = random.random()
            newSample[key] = d1[key] + gap*dif
        for key in d2.keys():
            if (not d1.has_key(key)):
                dif = d2[key]
                gap = random.random()
                newSample[key] = gap*dif
        newSamples.append(newSample)
    return newSamples

def smoteAlgo(dicts, rate, k, random_seed):
    # static seed for SMOTE algorithm
    random.seed(random_seed)
    n = len(dicts)
    print("[INFO] START SMOTE algo with rate = %s for %s samples" % (rate, n))
    
    nearestNeighbors = []
    newSamples = []
    for i in xrange(0, n):
        nearestNeighbors.append(PriorityQueue())

    for i, dict in enumerate(dicts):
        # if (i % 100 == 0):
        #     print("%s/%s" % (i, n))
        for j, dict2 in enumerate(dicts):
            if (i != j):
                nearestNeighbors[i].put((distance(dict, dict2), j))
                while (nearestNeighbors[i].qsize() > k):
                    nearestNeighbors[i].get()
        newSamples = newSamples + populateData(
            dicts,
            rate,
            dict,
            nearestNeighbors[i]
        )
    print("[INFO] DONE SMOTE algo, generate more %s" % (rate*n))
    return newSamples




