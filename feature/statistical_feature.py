#/usr/bin/env python
#-*-coding:utf-8-*-

"""
统计特征

编辑距离
杰拉德距离
"""

import codecs


class StatisticalFeature(object):
    def __init__(self, powerful_word_path):
        self.powerful_word_path = powerful_word_path
        self.powerful_word_dict = {}
        self._load_powerful_dict()


    def _load_powerful_dict(self, ):
        with codecs.open(self.powerful_word_path, encoding="utf-8") as f:
            for _, line in enumerate(f):
                line= line.strip("\n").strip()
                if not line:
                    continue
                line = line.split()
                if len(line)!=2:
                    continue
                w = line[0]
                p = line[1]
                self.powerful_word_dict[w] = float(p)


    def powerful_prob(self, text_1, text_2):
        join_words = set(text_1) | set(text_2)
        prob = 1.0
        for w in join_words:
            pw = self.powerful_word_dict.get(w, 0.5)
            prob *= pw
        return 1 - prob

    def same_words_rate(self, text_1, text_2):
        join_words = set(text_1) & set(text_2)
        total_len = len(text_1 + text_2)
        prob = 1.0 * (len(join_words)+1) / (total_len + 2)
        return prob


    def jaccard_distance(self, text_1, text_2):
        set_1 = set(text_1)
        set_2 = set(text_2)
        jaccard = 1.0*len(set_1.intersection(set_2))/len(set_1.union(set_2))
        return jaccard



if __name__ == '__main__':
    text_1 = [u'借款', u"花呗"]
    text_2 = [u'借款', u"花呗"]
    powerful_word_path = "../models/powerful_word.txt"
    statis_feature = StatisticalFeature(powerful_word_path)
    # jaccard = statis_feature.jaccard_distance(text_1, text_2)
    # print(jaccard)
    prob =  statis_feature.powerful_prob(text_1, text_2)
    print(prob)