#/usr/bin/env python
#-*-coding:utf-8-*-

#
import xgboost as xgb
import pandas as pd
import lsi_feature
import embedding_feature
import statistical_feature
import papre_process
import random
import codecs
import json
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics.classification import classification_report

def load_y(path):
    y =[]
    with open(path) as f:
        for _, line in enumerate(f):
            line = line.strip("\n").strip()
            line=line.split()
            label=int(line[0])
            y.append(label)
    return y

def load_feature_names():
    with codecs.open("feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    return feature_names


def train():
    # todo 自动参数选择
    feature_names = load_feature_names()
    dtrain = xgb.DMatrix('./xgboost.txt.train', feature_names=feature_names)
    dtest = xgb.DMatrix('./xgboost.txt.test', feature_names=feature_names)

    param = {'max_depth':10,
             'eta':0.06,
             'silent':1,
             'objective':'binary:logistic',
             'lambda':2 ,
             'subsample':0.8,
             'colsample_bytree':0.8,
             'verbose_eval':True,
             'seed':55
             }

    num_round = 4000

    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    y = load_y("./xgboost.txt.test")
    fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
    y_pred = [1 if p>0.5 else 0 for p in list(preds)]
    f1 = metrics.f1_score(y, y_pred)
    print("f1 socre; ", f1)
    print(classification_report(y, y_pred))
    bst.save_model("xgboost.model")

    feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, "#000099", label='ROC curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title("ROC")
    plt.show()


class TrainXgboost(object):
    def __init__(self, model="papre"):
        self.dictionary_path = "../models/gensim.dict"
        self.lsi_model_path = "../models/gensim.lsi"
        self.tfidf_model_path = "../models/gensim.tfidf"
        self.word2vec_model_path = "../models/gensim.word2vec"
        self.powerful_word_path = "../models/powerful_word.txt"

        self.train_file_path = "xgboost.txt.train"
        self.test_file_path = "xgboost.txt.train"

        self.lsi_feature_model = lsi_feature.LSIFeature(self.dictionary_path, self.tfidf_model_path, self.lsi_model_path)
        self.embedding_feature_model = embedding_feature.EmbeddingFeature(self.word2vec_model_path, self.tfidf_model_path, self.dictionary_path)
        self.static_feature_model = statistical_feature.StatisticalFeature(self.powerful_word_path)

        self.papre_processor = papre_process.PapreProcessor("../data/user_dict", "../data/stopwords.txt")

        self.train_outf = open("xgboost.txt.train", 'w')
        self.test_outf = open("xgboost.txt.test", 'w')

        self.train_file = "../data/atec_nlp_sim_train.csv"

    def create_train_data(self):
        all_data = []

        with codecs.open(self.train_file, encoding="utf-8") as f:
            for _, line in enumerate(f):
                line = line.strip("\n").strip()
                line = line.split("\t")
                text_1 = line[1]
                pre_res_1 = self.papre_processor.process(text_1)
                text_2 = line[2]
                pre_res_2 = self.papre_processor.process(text_2)
                label = line[3]
                if int(label) == 1:
                    for i in range(2):
                        all_data.append([pre_res_1, pre_res_2, label])
                else:
                    all_data.append([pre_res_1, pre_res_2, label])

        # shuffle
        random.shuffle(all_data)
        train_count = len(all_data) * 0.9


        feature_names = ["jaccard_dis", "lsi_dis", "tfidf_average_dis",  'same_words_rate', "cos_dis"]
        with codecs.open("feature_names.json", 'w', encoding="utf-8") as f:
            json.dump(feature_names, f)


        for i, item in enumerate(all_data):
            if i < train_count:
                outf = self.train_outf
            else:
                outf = self.test_outf
            words_1 = item[0].clean_words
            words_2 = item[1].clean_words
            label = item[2]

            jaccard_dis = self.static_feature_model.jaccard_distance(words_1, words_2)
            lsi_dis = self.lsi_feature_model.lsi_sim(words_1, words_2)
            tfidf_average_dis = self.embedding_feature_model.tfidf_average_dis(words_1, words_2)
            # deffent_prob = self.static_feature_model.powerful_prob(words_1, words_2)
            same_words_rate = self.static_feature_model.same_words_rate(words_1, words_2)
            # wmd_dis = self.embedding_feature_model.wmd_dis(words_1, words_2)
            cos_dis = self.embedding_feature_model.cos_dis(words_1, words_2)


            feature_list = [
                jaccard_dis,
                lsi_dis,
                tfidf_average_dis,
                same_words_rate,
                # wmd_dis,
                cos_dis
            ]
            self.write_line(label, feature_list, outf)
        self.train_outf.close()
        self.test_outf.close()

    def write_line(self, label, features, outf):
        feature_str = " ".join(["%s:%s"%(i, str(v),) for i,v in enumerate(features)])
        out_str = label+" "+feature_str
        outf.write(out_str+"\n")


if __name__ == '__main__':
    # train_xgboost = TrainXgboost()
    # train_xgboost.create_train_data()
    train()