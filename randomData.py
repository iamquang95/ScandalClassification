#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

import numpy
import codecs

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import KFold


import TextProcessUtils
import SMOTE

RANDOMSEED = 1508

def getMinoritySamples(dicts, labels):
    minoritySamples = []
    for dict, label in zip(dicts, labels):
        if (label == 1):
            minoritySamples.append(dict)
    return minoritySample

def getMajoritySamples(dicts, labels):
    minoritySamples = []
    for dict, label in zip(dicts, labels):
        if (label == 0):
            minoritySamples.append(dict)
    return minoritySample


if __name__ == "__main__":
    f = codecs.open("train.tagged", "r", "utf-8")

    data = [line for line in f]

    train_corpus = [TextProcessUtils.getTitle(line) for line in data]
    train_labels = [TextProcessUtils.getLabel(line) for line in data]

    f.close()

    f = codecs.open("test.tagged", "r", "utf-8")

    data = [line for line in f]

    test_corpus = [TextProcessUtils.getTitle(line) for line in data]
    test_labels = [TextProcessUtils.getLabel(line) for line in data]

    f.close()

    corpus = train_corpus + test_corpus
    labels = train_labels + test_labels

    totalLen = 0
    cntCorpus = 0

    for x in corpus:
        a = TextProcessUtils.lineToWords(x)
        totalLen += len(a)
        cntCorpus += 1
    avgLen = 1.0*totalLen/cntCorpus
    print(avgLen)

    # (corpus, labels) = shuffle(corpus, labels, random_state=RANDOMSEED)

    # kf = KFold(n_splits=10, random_state=RANDOMSEED)

    # for train_index, test_index in kf.split(corpus):
    #     test_corpus = []
    #     test_labels = []
    #     for index in test_index:
    #         test_corpus.append(corpus[index])
    #         test_labels.append(labels[index])
    #     f = codecs.open("validateData.txt", "w+", "utf-8")
    #     for (text, label) in zip(test_corpus, test_labels):
    #         f.write("%s %s\n" % (text, label))
    #     f.close()
    #     break



