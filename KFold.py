#!/usr/bin/python -tt
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys

import pickle

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
import UnderSampling
import OverSampling

positiveDict = TextProcessUtils.getDictionary("positive.dict")
negativeDict = TextProcessUtils.getDictionary("negative.dict")

RANDOMSEED = 1581995


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def getMinoritySamples(dicts, labels):
    minoritySamples = []
    for dict, label in zip(dicts, labels):
        if (label == 1):
            minoritySamples.append(dict)
    return minoritySamples

def features(tokenize, document):
    terms = tokenize(document)
    d = {
        'positive': TextProcessUtils.countWordInDict(positiveDict, document),
        'negative': TextProcessUtils.countWordInDict(negativeDict, document)
    }
    for t in terms:
        d[t] = d.get(t, 0) + 1
    return d


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

    (corpus, labels) = shuffle(corpus, labels, random_state=RANDOMSEED)

    kf = KFold(n_splits=5, random_state=RANDOMSEED)
    count_vectorizer = CountVectorizer(encoding=u'utf-8', ngram_range=(1, 3), max_df = 0.1, lowercase = True)
    tokenize = count_vectorizer.build_analyzer()

    kthRun = 0

    # gammaRange = [pow(2, x) for x in xrange(-9, -4)] #  [pow(2, x) for x in xrange(-10, -0)]
    # cRange = [pow(2, x) for x in xrange(0, 3)] #  [pow(2, x) for x in xrange(-3, 7)]
    # classWeightRange = [{1: pow(2, x)} for x in [0, 3]] + ['balanced'] #  [{1: pow(2, x)} for x in [0, 1, 2, 3]] + ['balanced']
    gammaRange = [0.00390625]
    cRange = [16]
    classWeightRange = ['balanced']
    tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': gammaRange,
            'C': cRange,
            'class_weight': classWeightRange,
            'decision_function_shape': ['ovr'] # ['ovo', 'ovr', None]
        }
    ]
    scores = ['f1'] # ['f1_macro', 'precision_macro', 'f1_micro']

    saved_best_param = {'kernel': 'rbf', 'C': 4, 'decision_function_shape': 'ovr', 'gamma': 0.00390625, 'class_weight': {1: 8}}
    saved_best_score = 0.0


    # for train_index, test_index in kf.split(corpus):
    #     kthRun += 1
    #     print(">>>>>>> Round = %s" % kthRun)
    #     eprint(">>>>>>> Round = %s" % kthRun)
    #     # init train data set
    #     train_corpus = []
    #     train_labels = []
    #     for index in train_index:
    #         train_corpus.append(corpus[index])
    #         train_labels.append(labels[index])
    #     # init test data set
    #     test_corpus = []
    #     test_labels = []
    #     for index in test_index:
    #         test_corpus.append(corpus[index])
    #         test_labels.append(labels[index])

    #     train_dict = [features(tokenize, d) for d in train_corpus]

    #     # SMOTE Algorithm
    #     # newMinoritySamples = SMOTE.smoteAlgo(
    #     #     getMinoritySamples(train_dict, train_labels),
    #     #     rate = 4,
    #     #     k = 100,
    #     #     random_seed = RANDOMSEED
    #     # )
    #     # train_dict = train_dict + newMinoritySamples
    #     # train_labels = train_labels + [1]*len(newMinoritySamples)

    #     # RandomUnderSampling Algorithm
    #     # train_dict, train_labels = UnderSampling.undersampling(train_dict, train_labels, RANDOMSEED)

    #     # RandomOverSampling Algorithm
    #     train_dict, train_labels = OverSampling.oversampling(train_dict, train_labels, RANDOMSEED)

    #     vect = DictVectorizer()
    #     train_counts = vect.fit_transform(train_dict)

    #     (train_counts, train_labels) = shuffle(train_counts, train_labels, random_state=RANDOMSEED)

    #     for score in scores:
    #         print("# Tuning hyper-parameters for %s" % score)
    #         print()

    #         clf = GridSearchCV(svm.SVC(C=1), tuned_parameters,
    #                         pre_dispatch='4*n_jobs',
    #                         n_jobs=4,
    #                         scoring=score,
    #                         verbose=1)
    #         clf.fit(train_counts, train_labels)

    #         print("Best parameters set found on development set of optimizing %s:" % (score))
    #         print()
    #         print(clf.best_params_)
    #         print()
    #         print("Grid scores on development set of optimizing %s:" % (score))
    #         print()
    #         means = clf.cv_results_['mean_test_score']
    #         stds = clf.cv_results_['std_test_score']
    #         for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #             print("%0.3f (+/-%0.03f) for %r"
    #                   % (mean, std * 2, params))
    #             if (params == clf.best_params_):
    #                 if (saved_best_score < mean):
    #                     saved_best_score = mean
    #                     saved_best_param = params
    #         print()

    #         print("Detailed classification report of optimizing %s:" % (score))
    #         print()
    #         print("The model is trained on the full development set.")
    #         print("The scores are computed on the full evaluation set.")
    #         print()
    #         # test_counts = count_vectorizer.transform(test_corpus)
    #         test_counts = vect.transform(features(tokenize, d) for d in test_corpus)
    #         y_true, y_pred = test_labels, clf.predict(test_counts)
    #         print(metrics.classification_report(y_true, y_pred))
    #         print()
    #         print(metrics.confusion_matrix(y_true, y_pred))


    #         eprint(metrics.classification_report(y_true, y_pred))
    #         eprint()
    #         eprint(metrics.confusion_matrix(y_true, y_pred))

    print(saved_best_score)
    print(saved_best_param)

    train_dict = [features(tokenize, d) for d in corpus]

    # SMOTE
    # newMinoritySamples = SMOTE.smoteAlgo(
    #     getMinoritySamples(train_dict, train_labels),
    #     rate = 4,
    #     k = 100,
    #     random_seed = RANDOMSEED
    #     )
    # train_dict = train_dict + newMinoritySamples
    # train_labels = labels + [1]*len(newMinoritySamples)

    # Random Under_sampling
    train_dict, train_labels = UnderSampling.undersampling(train_dict, labels, RANDOMSEED)

    # Random Over_sampling

    # train_dict, train_labels = OverSampling.oversampling(train_dict, labels, RANDOMSEED)

    vect = DictVectorizer()
    train_counts = vect.fit_transform(train_dict)

    vector_model = "vector_random_us.sav"
    pickle.dump(vect, open(vector_model, 'wb'))

    print("[INFO] FINISH creating data for learning")

    model = svm.SVC(C = saved_best_param['C'],
                    kernel = saved_best_param['kernel'],
                    decision_function_shape = 'ovr',
                    gamma = saved_best_param['gamma'],
                    class_weight = saved_best_param['class_weight'],
                    probability=True
        )
    model.fit(train_counts, train_labels)

    print("[INFO] FINISH training model")

    model_file_name = "model_random_us.sav"
    pickle.dump(model, open(model_file_name, 'wb'))

    # f = codecs.open("TokedData.txt", "r", "utf-8")
    # unlabeled_data = [line for line in f]
    # f.close()

    # print("[INFO] FINISH reading un-labeled data")

    # newPos = codecs.open("newPos.txt", "w+", "utf-8")
    # newNeg = codecs.open("newNeg.txt", "w+", "utf-8")
    # notPredict = codecs.open("TokedData.txt", "w+", "utf-8")

    # unlabeled_data_counts = vect.transform(features(tokenize, d) for d in unlabeled_data)

    # predicted_value = model.predict_proba(unlabeled_data_counts)

    # print("[INFO] FINISH predicting un-labeled data")

    # for (probs, data) in zip(predicted_value, unlabeled_data):
    #     data = TextProcessUtils.removeEndline(data)
    #     if (probs[0] > 0.98):
    #         newNeg.write("%s %s\n" % (data, "0"))
    #     elif (probs[1] > 0.98):
    #         newPos.write("%s %s\n" % (data, "1"))
    #     else:
    #         notPredict.write("%s\n" % (data))

    # newPos.close()
    # newNeg.close()
    # notPredict.close()










