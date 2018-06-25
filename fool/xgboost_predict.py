#/usr/bin/env python
#-*-coding:utf-8-*-

import xgboost as xgb


import lsi_feature
import embedding_feature
import statistical_feature
import papre_process
import codecs
import os
import sys
import json

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
def load_feature_names(path):
    with codecs.open(path, encoding="utf-8") as f:
        feature_names = json.load(f)
    return feature_names

class Predictor():
    def __init__(self):
        self.dictionary_path = os.path.join(CURRENT_PATH,"./models/gensim.dict")
        self.lsi_model_path = os.path.join(CURRENT_PATH,"./models/gensim.lsi")
        self.tfidf_model_path = os.path.join(CURRENT_PATH,"./models/gensim.tfidf")
        self.word2vec_model_path = os.path.join(CURRENT_PATH,"./models/gensim.word2vec")
        self.feature_names_path = os.path.join(CURRENT_PATH, "./models/feature_names.json")
        self.powerful_word_path = os.path.join(CURRENT_PATH, "./models/powerful_word.txt")


        self.feature_names = load_feature_names(self.feature_names_path)
        self.lsi_feature_model = lsi_feature.LSIFeature(self.dictionary_path, self.tfidf_model_path, self.lsi_model_path)
        self.embedding_feature_model = embedding_feature.EmbeddingFeature(self.word2vec_model_path, self.tfidf_model_path, self.dictionary_path)
        self.static_feature_model = statistical_feature.StatisticalFeature(self.powerful_word_path)
        self.papre_processor = papre_process.PapreProcessor(os.path.join(CURRENT_PATH,"./models/user_dict"), os.path.join(CURRENT_PATH,"./models/stopwords.txt"))
        param = {'max_depth':10,
             'eta':0.1,
             'silent':1,
             'objective':'binary:logistic',
             'lambda':2 ,
             'subsample':0.8,
             'colsample_bytree':0.8,
             'verbose_eval':True,
             'seed':55
             }
        self.model = xgb.Booster(param)
        self.model.load_model(fname=os.path.join(CURRENT_PATH, "./models/xgboost.model"))



    def predict(self,text_1, text_2):
        pre_res_1 = self.papre_processor.process(text_1)
        pre_res_2 = self.papre_processor.process(text_2)
        words_1 = pre_res_1.clean_words
        words_2 = pre_res_2.clean_words

        jaccard_dis = self.static_feature_model.jaccard_distance(words_1, words_2)
        lsi_dis = self.lsi_feature_model.lsi_sim(words_1, words_2)
        tfidf_average_dis = self.embedding_feature_model.tfidf_average_dis(words_1, words_2)
        # deffent_prob = self.static_feature_model.powerful_prob(words_1, words_2)
        same_words_rate = self.static_feature_model.same_words_rate(words_1, words_2)
        # wmd_dis = self.embedding_feature_model.wmd_dis(words_1, words_2)
        cos_dis = self.embedding_feature_model.cos_dis(text_1, text_2)

        feature_list = [
            jaccard_dis,
            lsi_dis,
            tfidf_average_dis,
            # deffent_prob,
            same_words_rate,
            # wmd_dis
            cos_dis
        ]

        data = xgb.DMatrix(feature_list, feature_names=self.feature_names)
        pred = self.model.predict(data)[0]
        preds_label = 1 if pred > 0.5 else 0
        return preds_label, pred

def process(inpath, outpath):
    predictor = Predictor()

    with codecs.open(inpath, 'r', encoding="utf-8") as fin, codecs.open(outpath, 'w', encoding="utf-8") as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split("\t")
            pred_label, prob = predictor.predict(sen1, sen2)
            fout.write(lineno + '\t%d\n' % (pred_label))


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])

