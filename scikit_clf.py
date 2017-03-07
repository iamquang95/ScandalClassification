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


import TextProcessUtils
import SMOTE

positiveDict = TextProcessUtils.getDictionary("positive.dict")
negativeDict = TextProcessUtils.getDictionary("negative.dict")

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

def shuffleSamples(counts, labels):
    return shuffle(counts, labels)

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

    (corpus, labels) = shuffle(corpus, labels, random_state=158)

    train_corpus, test_corpus, train_labels, test_labels = train_test_split(
        corpus, labels, test_size=0.1, random_state=158)

    # NORMAL USE
    count_vectorizer = CountVectorizer(encoding=u'utf-8', ngram_range=(1, 3), max_df = 0.1, lowercase = True)
    tokenize = count_vectorizer.build_analyzer()
    counts = count_vectorizer.fit_transform(train_corpus)

    # classifier = MultinomialNB()
    # classifier.fit(counts, train_labels)

    examples = [u'Những hình_ảnh “ ” cổng trường_học Với học_sinh sử_dụng đồ_ăn bẩn cổng trường_học hiện_nay tích_lũy 20 30 con_số mắc bệnh ung_thư tiểu_đường tim_mạch gia_tăng Những_hình_ảnh hình_ảnh_“ “_không không_muốn muốn_thấy thấy_” ”_trước trước_cổng cổng_trường_học trường_học_Với Với_việc việc_học_sinh học_sinh_sử_dụng sử_dụng_đồ_ăn đồ_ăn_bẩn bẩn_trước trước_cổng cổng_trường_học trường_học_như như_hiện_nay hiện_nay_tích_lũy tích_lũy_dần_dần dần_dần_đến đến_20 20_hoặc hoặc_30 30_năm năm_nữa nữa_con_số con_số_mắc mắc_bệnh bệnh_ung_thư ung_thư_tiểu_đường tiểu_đường_tim_mạch tim_mạch_sẽ sẽ_gia_tăng gia_tăng_rất rất_nhiều Những_Những_“ hình_ảnh_hình_ảnh_không “_“_muốn không_không_thấy muốn_muốn_” thấy_thấy_trước ”_”_cổng trước_trước_trường_học cổng_cổng_Với trường_học_trường_học_việc Với_Với_học_sinh việc_việc_sử_dụng học_sinh_học_sinh_đồ_ăn sử_dụng_sử_dụng_bẩn đồ_ăn_đồ_ăn_trước bẩn_bẩn_cổng trước_trước_trường_học cổng_cổng_như trường_học_trường_học_hiện_nay như_như_tích_lũy hiện_nay_hiện_nay_dần_dần tích_lũy_tích_lũy_đến dần_dần_dần_dần_20 đến_đến_hoặc 20_20_30 hoặc_hoặc_năm 30_30_nữa năm_năm_con_số nữa_nữa_mắc con_số_con_số_bệnh mắc_mắc_ung_thư bệnh_bệnh_tiểu_đường ung_thư_ung_thư_tim_mạch tiểu_đường_tiểu_đường_sẽ tim_mạch_tim_mạch_gia_tăng sẽ_sẽ_rất gia_tăng_gia_tăng_nhiều',
                u'Điều_kiện kinh_doanh dịch_vụ đào_tạo lái_xe ô_tô và dịch_vụ sát_hạch lái_xe . Ngày 1/7 , Nghị_định số 65/2016 / NĐ - CP quy_định về điều_kiện kinh_doanh dịch_vụ đào_tạo lái_xe ô_tô và dịch_vụ sát_hạch lái_xe chính_thức có hiệu_lực thi_hành .',
                u'Thí_sinh bị “ tra_tấn ” bởi tiếng ồn từ công_trình xây_dựng .',
                u'Xem_xét buộc thôi_việc thầy_giáo đổ nước vô miệng học_sinh . Một thầy_giáo ở Bình_Định buộc học_sinh nằm ngửa trên bục giảng rồi lấy nước đổ liên_tiếp vào miệng chỉ vì học_sinh này nhắc các bạn im_lặng khi lớp_học ồn_ào .',
                u'HS phạm Luật_Giao thông : Cần tăng_cường giáo_dục . Đã có thêm nhiều ý_kiến bàn_luận xung_quanh mức chế_tài học_sinh vi_phạm Luật giao_thông do Sở GD - ĐT Hà_Nội vừa ban_hành ( Tuổi_Trẻ ngày 9 và 10-3 ) .',
                u'Giáo_viên bắt học_sinh " khỏa_thân " trước cửa lớp vì không hoàn_thành bài_tập gây phẫn_nộ . Hình_ảnh hai học_sinh bị giáo_viên lột trần_truồng đứng bên ngoài cửa lớp để trừng_phạt vì tội không làm bài_tập về nhà khiến dư_luận vô_cùng bức_xúc .',
                u'Thu về hơn 2.900 đơn_vị máu tình_nguyện vì đồng_đội thân_yêu . Chiều 8-1 , Ngày hội hiến máu tình_nguyện “ Giọt máu nghĩa_tình vì đồng_đội thân_yêu ” - Lần thứ II và Liên_hoan dân vũ sinh_viên các Học_viện , trường CAND mở_rộng năm 2016 đã bế_mạc tại Trung_tâm Thể_thao CAND ( Nguyễn_Xiển , Thanh_Trì , Hà_Nội )']
    example_counts = count_vectorizer.transform(examples)

    print("[INFO] Finished read data")

    # predictions = classifier.predict(example_counts)
    # print(predictions) # should be [1, 0, 1, 1, 0, 1, 0]

    # PIPELINE

    # pipeline = Pipeline([
    #       ('vectorizer',  CountVectorizer(encoding=u'utf-8')),
    #       ('classifier',  svm.SVC())
    #     ])
    # pipeline.fit(train_corpus, train_corpus)
    # print(pipeline.predict(examples)) # should be [1, 0, 1, 1, 0, 1, 0]

    # predicted = pipeline.predict(test_corpus)
    # print(numpy.mean(predicted == test_labels))

    # print(metrics.classification_report(test_labels, predicted,
    #   target_names=['0', '1']))

    # GridSearch to find best parameter
    gammaRange = [pow(2, x) for x in xrange(-10, -0)]
    cRange =  [pow(2, x) for x in xrange(-3, 7)]
    classWeightRange = [{1: pow(2, x)} for x in [0, 1, 2, 3]] + ['balanced']
    tuned_parameters = [
        {
            'kernel': ['rbf'],
            'gamma': gammaRange,
            'C': cRange,
            'class_weight': classWeightRange,
            'decision_function_shape': ['ovr'] # ['ovo', 'ovr', None]
        }
    ]
    scores = ['f1_macro', 'precision_macro', 'f1_micro']

    # train_counts = count_vectorizer.fit_transform(train_corpus)
    # vect = DictVectorizer()
    # train_counts = vect.fit_transform(features(tokenize, d) for d in train_corpus)

    train_dict = [features(tokenize, d) for d in train_corpus]

    newMinoritySamples = SMOTE.smoteAlgo(
        getMinoritySamples(train_dict, train_labels),
        rate = 5,
        k = 20
    )

    train_dict = train_dict + newMinoritySamples
    train_labels = train_labels + [1]*len(newMinoritySamples)

    vect = DictVectorizer()
    train_counts = vect.fit_transform(train_dict)

    (train_counts, train_labels) = shuffle(train_counts, train_labels)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv = 5,
                        pre_dispatch='4*n_jobs',
                        n_jobs=-1,
                        scoring=score,
                        verbose=10)
        clf.fit(train_counts, train_labels)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
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

    # # K_FOLD
    # k_fold = KFold(n=len(train_corpus), n_folds=5)
    # scores = []
    # confusion = numpy.array([[0, 0], [0, 0]])
    # for train_indices, test_indices in k_fold:
    #     train_text = [train_corpus[index] for index in train_indices]
    #     train_y = [train_labels[index] for index in train_indices]

    #     test_text = [train_corpus[index] for index in test_indices]
    #     test_y = [train_labels[index] for index in test_indices]

    #     pipeline.fit(train_text, train_y)
    #     predictions = pipeline.predict(test_text)

    #     confusion += confusion_matrix(test_y, predictions)
    #     score = f1_score(test_y, predictions, pos_label=u'1')
    #     scores.append(score)

    # print('Total articles classified:', len(data))
    # print('Score:', sum(scores)/len(scores))
    # print('Confusion matrix:')
    # print(confusion)


