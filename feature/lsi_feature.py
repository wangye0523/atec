#/usr/bin/env python
#-*-coding:utf-8-*-


from gensim import corpora
from gensim.models import TfidfModel, LsiModel
import numpy as np
from scipy import linalg

class LSIFeature(object):
    def __init__(self, dictionary_path, tfidf_model_path, lsi_model_path):
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.tfidf_model = TfidfModel.load(tfidf_model_path)
        self.lsi_model = LsiModel.load(lsi_model_path)

    def doc_to_vec(self, text):
        doc = self.dictionary.doc2bow(text)
        tfidf_vec = self.tfidf_model[doc]
        lsi_vec = self.lsi_model[tfidf_vec]
        return lsi_vec

    def lsi_sim(self, text_1, text_2):
        vec_1 = np.array([item[1] for item in self.doc_to_vec(text_1)])
        vec_2 = np.array([item[1] for item in self.doc_to_vec(text_2)])
        try:
            if len(vec_1) != len(vec_2):
                return 0
            num = np.dot(vec_1.T, vec_2)
            denom = linalg.norm(vec_1) * linalg.norm(vec_2)
            cos = num / denom
            sim = 0.5 + 0.5 * cos
        except Exception as e:
            print(e)
            sim = 0.5
        return sim


if __name__ == '__main__':
    dictionary_path = "../models/gensim.dict"
    tfidf_model_path = "../models/gensim.tfidf"
    lsi_model_path = "../modelsf/gensim.lsi"

    text_1 = [u"还款", u"花呗"]
    text_2 = [u"开通", u"借呗"]
    lsi_feature = LSIFeature(dictionary_path, tfidf_model_path, lsi_model_path)
    lsi_dis = lsi_feature.lsi_sim(text_1, text_2)
    print(lsi_dis)


