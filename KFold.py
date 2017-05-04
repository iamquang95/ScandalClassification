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
import UnderSampling
import OverSampling

positiveDict = TextProcessUtils.getDictionary("positive.dict")
negativeDict = TextProcessUtils.getDictionary("negative.dict")

RANDOMSEED = 1581995

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

    gammaRange = [pow(2, x) for x in xrange(-9, -4)] #  [pow(2, x) for x in xrange(-10, -0)]
    cRange = [pow(2, x) for x in xrange(0, 3)] #  [pow(2, x) for x in xrange(-3, 7)]
    classWeightRange = [{1: pow(2, x)} for x in [0, 3]] + ['balanced'] #  [{1: pow(2, x)} for x in [0, 1, 2, 3]] + ['balanced']
    # gammaRange = [0.00390625]
    # cRange = [16]
    # classWeightRange = ['balanced']
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

    saved_best_param = {'kernel': 'rbf', 'C': 16, 'decision_function_shape': 'ovr', 'gamma': 0.00390625, 'class_weight': {1: 1}}
    saved_best_score = 0.0


    for train_index, test_index in kf.split(corpus):
        kthRun += 1
        print(">>>>>>> Round = %s" % kthRun)
        # init train data set
        train_corpus = []
        train_labels = []
        for index in train_index:
            train_corpus.append(corpus[index])
            train_labels.append(labels[index])
        # init test data set
        test_corpus = []
        test_labels = []
        for index in test_index:
            test_corpus.append(corpus[index])
            test_labels.append(labels[index])

        train_dict = [features(tokenize, d) for d in train_corpus]

        # SMOTE Algorithm
        # newMinoritySamples = SMOTE.smoteAlgo(
        #     getMinoritySamples(train_dict, train_labels),
        #     rate = 4,
        #     k = 100,
        #     random_seed = RANDOMSEED
        # )
        # train_dict = train_dict + newMinoritySamples
        # train_labels = train_labels + [1]*len(newMinoritySamples)

        # RandomUnderSampling Algorithm
        # train_dict, train_labels = UnderSampling.undersampling(train_dict, train_labels, RANDOMSEED)

        # RandomOverSampling Algorithm
        train_dict, train_labels = OverSampling.oversampling(train_dict, train_labels, RANDOMSEED)

        vect = DictVectorizer()
        train_counts = vect.fit_transform(train_dict)

        (train_counts, train_labels) = shuffle(train_counts, train_labels, random_state=RANDOMSEED)

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters,
                            pre_dispatch='4*n_jobs',
                            n_jobs=6,
                            scoring=score,
                            verbose=1)
            clf.fit(train_counts, train_labels)

            print("Best parameters set found on development set of optimizing %s:" % (score))
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set of optimizing %s:" % (score))
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
                if (params == clf.best_params_):
                    if (saved_best_score < mean):
                        saved_best_score = mean
                        saved_best_param = params
            print()

            print("Detailed classification report of optimizing %s:" % (score))
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            # test_counts = count_vectorizer.transform(test_corpus)
            test_counts = vect.transform(features(tokenize, d) for d in test_corpus)
            y_true, y_pred = test_labels, clf.predict(test_counts)
            print(metrics.classification_report(y_true, y_pred))
            print()
            print(metrics.confusion_matrix(y_true, y_pred))

    # print(saved_best_score)
    # print(saved_best_param)

    # train_dict = [features(tokenize, d) for d in corpus]
    # newMinoritySamples = SMOTE.smoteAlgo(
    #     getMinoritySamples(train_dict, train_labels),
    #     rate = 4,
    #     k = 100,
    #     random_seed = RANDOMSEED
    #     )
    # train_dict = train_dict + newMinoritySamples
    # train_labels = labels + [1]*len(newMinoritySamples)

    # vect = DictVectorizer()
    # train_counts = vect.fit_transform(train_dict)

    # print("[INFO] FINISH creating data for learning")

    # model = svm.SVC(C = saved_best_param['C'],
    #                 kernel = saved_best_param['kernel'],
    #                 decision_function_shape = 'ovr',
    #                 gamma = saved_best_param['gamma'],
    #                 class_weight = saved_best_param['class_weight'],
    #                 probability=True
    #     )
    # model.fit(train_counts, train_labels)

    # print("[INFO] FINISH training model")

    # # examples = [u'Những hình_ảnh “ ” cổng trường_học Với học_sinh sử_dụng đồ_ăn bẩn cổng trường_học hiện_nay tích_lũy 20 30 con_số mắc bệnh ung_thư tiểu_đường tim_mạch gia_tăng Những_hình_ảnh hình_ảnh_“ “_không không_muốn muốn_thấy thấy_” ”_trước trước_cổng cổng_trường_học trường_học_Với Với_việc việc_học_sinh học_sinh_sử_dụng sử_dụng_đồ_ăn đồ_ăn_bẩn bẩn_trước trước_cổng cổng_trường_học trường_học_như như_hiện_nay hiện_nay_tích_lũy tích_lũy_dần_dần dần_dần_đến đến_20 20_hoặc hoặc_30 30_năm năm_nữa nữa_con_số con_số_mắc mắc_bệnh bệnh_ung_thư ung_thư_tiểu_đường tiểu_đường_tim_mạch tim_mạch_sẽ sẽ_gia_tăng gia_tăng_rất rất_nhiều Những_Những_“ hình_ảnh_hình_ảnh_không “_“_muốn không_không_thấy muốn_muốn_” thấy_thấy_trước ”_”_cổng trước_trước_trường_học cổng_cổng_Với trường_học_trường_học_việc Với_Với_học_sinh việc_việc_sử_dụng học_sinh_học_sinh_đồ_ăn sử_dụng_sử_dụng_bẩn đồ_ăn_đồ_ăn_trước bẩn_bẩn_cổng trước_trước_trường_học cổng_cổng_như trường_học_trường_học_hiện_nay như_như_tích_lũy hiện_nay_hiện_nay_dần_dần tích_lũy_tích_lũy_đến dần_dần_dần_dần_20 đến_đến_hoặc 20_20_30 hoặc_hoặc_năm 30_30_nữa năm_năm_con_số nữa_nữa_mắc con_số_con_số_bệnh mắc_mắc_ung_thư bệnh_bệnh_tiểu_đường ung_thư_ung_thư_tim_mạch tiểu_đường_tiểu_đường_sẽ tim_mạch_tim_mạch_gia_tăng sẽ_sẽ_rất gia_tăng_gia_tăng_nhiều',
    # #             u'Điều_kiện kinh_doanh dịch_vụ đào_tạo lái_xe ô_tô và dịch_vụ sát_hạch lái_xe . Ngày 1/7 , Nghị_định số 65/2016 / NĐ - CP quy_định về điều_kiện kinh_doanh dịch_vụ đào_tạo lái_xe ô_tô và dịch_vụ sát_hạch lái_xe chính_thức có hiệu_lực thi_hành .',
    # #             u'Thí_sinh bị “ tra_tấn ” bởi tiếng ồn từ công_trình xây_dựng .',
    # #             u'Xem_xét buộc thôi_việc thầy_giáo đổ nước vô miệng học_sinh . Một thầy_giáo ở Bình_Định buộc học_sinh nằm ngửa trên bục giảng rồi lấy nước đổ liên_tiếp vào miệng chỉ vì học_sinh này nhắc các bạn im_lặng khi lớp_học ồn_ào .',
    # #             u'HS phạm Luật_Giao thông : Cần tăng_cường giáo_dục . Đã có thêm nhiều ý_kiến bàn_luận xung_quanh mức chế_tài học_sinh vi_phạm Luật giao_thông do Sở GD - ĐT Hà_Nội vừa ban_hành ( Tuổi_Trẻ ngày 9 và 10-3 ) .',
    # #             u'Giáo_viên bắt học_sinh " khỏa_thân " trước cửa lớp vì không hoàn_thành bài_tập gây phẫn_nộ . Hình_ảnh hai học_sinh bị giáo_viên lột trần_truồng đứng bên ngoài cửa lớp để trừng_phạt vì tội không làm bài_tập về nhà khiến dư_luận vô_cùng bức_xúc .',
    # #             u'Thu về hơn 2.900 đơn_vị máu tình_nguyện vì đồng_đội thân_yêu . Chiều 8-1 , Ngày hội hiến máu tình_nguyện “ Giọt máu nghĩa_tình vì đồng_đội thân_yêu ” - Lần thứ II và Liên_hoan dân vũ sinh_viên các Học_viện , trường CAND mở_rộng năm 2016 đã bế_mạc tại Trung_tâm Thể_thao CAND ( Nguyễn_Xiển , Thanh_Trì , Hà_Nội )']
    # # example_counts = vect.transform(features(tokenize, d) for d in examples)

    # # class_probabilities = model.predict_proba(example_counts)
    # # print(class_probabilities)

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










