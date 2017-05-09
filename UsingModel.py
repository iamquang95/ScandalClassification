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

# vector_file_name = "vector_random_os.sav"
# model_file_name = "model_random_os.sav"

vector_file_name = "vector_random_us.sav"
model_file_name = "model_random_us.sav"

# vector_file_name = "vector_smote.sav"
# model_file_name = "model_smote.sav"

positiveDict = TextProcessUtils.getDictionary("positive.dict")
negativeDict = TextProcessUtils.getDictionary("negative.dict")


def features(tokenize, document):
    terms = tokenize(document)
    d = {
        'positive': TextProcessUtils.countWordInDict(positiveDict, document),
        'negative': TextProcessUtils.countWordInDict(negativeDict, document)
    }
    for t in terms:
        d[t] = d.get(t, 0) + 1
    return d

count_vectorizer = CountVectorizer(encoding=u'utf-8', ngram_range=(1, 3), max_df = 0.1, lowercase = True)
tokenize = count_vectorizer.build_analyzer()

# Load vector
vect = pickle.load(open(vector_file_name, 'rb'))

# Load model

model = pickle.load(open(model_file_name, 'rb'))

examples = [u'Những hình_ảnh “ ” cổng trường_học Với học_sinh sử_dụng đồ_ăn bẩn cổng trường_học hiện_nay tích_lũy 20 30 con_số mắc bệnh ung_thư tiểu_đường tim_mạch gia_tăng Những_hình_ảnh hình_ảnh_“ “_không không_muốn muốn_thấy thấy_” ”_trước trước_cổng cổng_trường_học trường_học_Với Với_việc việc_học_sinh học_sinh_sử_dụng sử_dụng_đồ_ăn đồ_ăn_bẩn bẩn_trước trước_cổng cổng_trường_học trường_học_như như_hiện_nay hiện_nay_tích_lũy tích_lũy_dần_dần dần_dần_đến đến_20 20_hoặc hoặc_30 30_năm năm_nữa nữa_con_số con_số_mắc mắc_bệnh bệnh_ung_thư ung_thư_tiểu_đường tiểu_đường_tim_mạch tim_mạch_sẽ sẽ_gia_tăng gia_tăng_rất rất_nhiều Những_Những_“ hình_ảnh_hình_ảnh_không “_“_muốn không_không_thấy muốn_muốn_” thấy_thấy_trước ”_”_cổng trước_trước_trường_học cổng_cổng_Với trường_học_trường_học_việc Với_Với_học_sinh việc_việc_sử_dụng học_sinh_học_sinh_đồ_ăn sử_dụng_sử_dụng_bẩn đồ_ăn_đồ_ăn_trước bẩn_bẩn_cổng trước_trước_trường_học cổng_cổng_như trường_học_trường_học_hiện_nay như_như_tích_lũy hiện_nay_hiện_nay_dần_dần tích_lũy_tích_lũy_đến dần_dần_dần_dần_20 đến_đến_hoặc 20_20_30 hoặc_hoặc_năm 30_30_nữa năm_năm_con_số nữa_nữa_mắc con_số_con_số_bệnh mắc_mắc_ung_thư bệnh_bệnh_tiểu_đường ung_thư_ung_thư_tim_mạch tiểu_đường_tiểu_đường_sẽ tim_mạch_tim_mạch_gia_tăng sẽ_sẽ_rất gia_tăng_gia_tăng_nhiều',
            u'Điều_kiện kinh_doanh dịch_vụ đào_tạo lái_xe ô_tô và dịch_vụ sát_hạch lái_xe . Ngày 1/7 , Nghị_định số 65/2016 / NĐ - CP quy_định về điều_kiện kinh_doanh dịch_vụ đào_tạo lái_xe ô_tô và dịch_vụ sát_hạch lái_xe chính_thức có hiệu_lực thi_hành .',
            u'Bí_thư Đoàn nhiệt_huyết . ( HNM ) - Đảng_viên trẻ , Bí_thư Đoàn phường Hàng_Bồ ( quận Hoàn_Kiếm ) Nguyễn_Phú_Ngọc được đánh_giá là một thanh_niên giàu nhiệt_huyết , trách_nhiệm và năng_động để đem lại lợi_ích thiết_thực cho cộng_đồng .',
            u'Xem_xét buộc thôi_việc thầy_giáo đổ nước vô miệng học_sinh . Một thầy_giáo ở Bình_Định buộc học_sinh nằm ngửa trên bục giảng rồi lấy nước đổ liên_tiếp vào miệng chỉ vì học_sinh này nhắc các bạn im_lặng khi lớp_học ồn_ào .',
            u'HS phạm Luật_Giao thông : Cần tăng_cường giáo_dục . Đã có thêm nhiều ý_kiến bàn_luận xung_quanh mức chế_tài học_sinh vi_phạm Luật giao_thông do Sở GD - ĐT Hà_Nội vừa ban_hành ( Tuổi_Trẻ ngày 9 và 10-3 ) .',
            u'Giáo_viên bắt học_sinh " khỏa_thân " trước cửa lớp vì không hoàn_thành bài_tập gây phẫn_nộ . Hình_ảnh hai học_sinh bị giáo_viên lột trần_truồng đứng bên ngoài cửa lớp để trừng_phạt vì tội không làm bài_tập về nhà khiến dư_luận vô_cùng bức_xúc .',
            u'Thu về hơn 2.900 đơn_vị máu tình_nguyện vì đồng_đội thân_yêu . Chiều 8-1 , Ngày hội hiến máu tình_nguyện “ Giọt máu nghĩa_tình vì đồng_đội thân_yêu ” - Lần thứ II và Liên_hoan dân vũ sinh_viên các Học_viện , trường CAND mở_rộng năm 2016 đã bế_mạc tại Trung_tâm Thể_thao CAND ( Nguyễn_Xiển , Thanh_Trì , Hà_Nội )',
            u'Vụ 50.000 thẻ BHYT cùng \' sinh_nhật \' : Xin_lỗi học_sinh , cấp lại thẻ . Bảo_hiểm_xã_hội TP Buôn_Ma_Thuột vừa xin_lỗi gần 1.000 học_sinh , giáo_viên của Trường THCS Lương_Thế_Vinh vì đã cấp thẻ bảo_hiểm_y_tế cho học_sinh có cùng " ngày sinh_nhật " .']

example_counts = vect.transform(features(tokenize, d) for d in examples)

class_probabilities = model.predict_proba(example_counts)

def betterShowingResult(samples, probs):
	for (sample, prob) in zip(samples, probs):
		p1 = prob[0] * 100
		p2 = prob[1] * 100
		label = "Normal"
		if (p2 > p1):
			label = "Scandal"

		# print(sample)
		print("%s %s" % (format(p1, '.2f'), format(p2, '.2f')))
		print(label)
		print("---------------------------------")


# print(class_probabilities)
betterShowingResult(examples, class_probabilities)










