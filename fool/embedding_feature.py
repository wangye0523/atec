#/usr/bin/env python
#-*-coding:utf-8-*-

"""
embedding feature
wmd
"""

from gensim.models import  KeyedVectors, TfidfModel, Word2Vec
from gensim import corpora
import numpy as np
from scipy import linalg

class EmbeddingFeature(object):
    def __init__(self, word2vec_model_path, tfidf_model_path, dictionary_path):
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        self.tfidf_model = TfidfModel.load(tfidf_model_path)
        self.dictionary = corpora.Dictionary.load(dictionary_path)


    def wmd_dis(self, text_1, text_2):
        wmddis = self.word2vec_model.wv.wmdistance(text_1, text_2)
        return wmddis

    def doc_to_vec(self, text):
        doc = self.dictionary.doc2bow(text)
        tfidf_vec = self.tfidf_model[doc]
        doc_vec = np.array([0.0]*self.word2vec_model.vector_size)
        word_counter = 0
        for wid, tfidf in tfidf_vec:
            w = self.dictionary[wid]
            if w not in self.word2vec_model.wv:
                continue
            embed_vec = self.word2vec_model.wv.word_vec(w)
            doc_vec += tfidf * embed_vec
            word_counter += 1
        return doc_vec, word_counter


    def tfidf_average_dis(self, text_1, text_2):
        try:
            vec_1, word_count_1 = self.doc_to_vec(text_1)
            vec_2, word_count_2 = self.doc_to_vec(text_2)
            if not all([word_count_1, word_count_2]):
                return 0.0
            num = np.dot(vec_1.T, vec_2)
            denom = linalg.norm(vec_1) * linalg.norm(vec_2)
            cos = num / denom
            sim = 0.5 + 0.5 * cos
        except Exception as e:
            print(e)
            sim = 0.5
        return sim


    def cos_dis(self,text_1, text_2):
        text_1 = [w for w in text_1 if w in self.word2vec_model.wv]
        text_2 = [w for w in text_2 if w in self.word2vec_model.wv]
        if not(len(text_1) and len(text_2)):
            return 0.5
        dis = self.word2vec_model.wv.n_similarity(text_1, text_2)
        return dis



if __name__ == '__main__':
    text_1 = [u"还款", u"花呗" ]
    text_2 = [u"还款", u"借呗"]
    dictionary_path = "../models/gensim.dict"
    tfidf_model_path = "../models/gensim.tfidf"
    lsi_model_path = "../models/gensim.lsi"

    embed_feature = EmbeddingFeature("../models/gensim.word2vec", tfidf_model_path, dictionary_path)
    # dis =  embed_feature.wmd_dis(text_1, text_2)
    tfidf_dis = embed_feature.tfidf_average_dis(text_1, text_2)
    cos_dis = embed_feature.cos_dis(text_1, text_2)
    wmd_dis = embed_feature.wmd_dis(text_1, text_2)
    print(tfidf_dis)
    print(cos_dis)
    print(wmd_dis)



