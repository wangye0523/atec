#/usr/bin/env python
#-*-coding:utf-8-*-

"""
统计每个词对句子是否相同的贡献度
"""

#

SOURCE_PATH = "../data/atec_nlp_sim_train.csv"
from papre_process import PapreProcessor


class TrainStatisFeature():
    def __init__(self):
        self.papre_processor = PapreProcessor("../data/user_dict", "../data/stopwords.txt")
        self.powerful__dict = {}


    def train_powerful_word(self):
        """
        统计每个词对句子相同的贡献读
        :return:
        """
        with open(SOURCE_PATH) as f:
            for _, line in enumerate(f):
                line = line.strip("\n").strip()
                line = line.split("\t")
                text_1 = line[1]
                text_2 = line[2]
                pre_1 = self.papre_processor.process(text_1)
                pre_2 = self.papre_processor.process(text_2)
                clean_words_1 = pre_1.clean_words
                clean_words_2 = pre_2.clean_words
                label = line[-1]
                # 求交集
                join_words = set(clean_words_1) & set(clean_words_2)
                for w in join_words:
                    if w not in self.powerful__dict:
                        self.powerful__dict[w] = {0:0, 1:0}
                    self.powerful__dict[w][int(label)] += 1

        model_outf = open("../models/powerful_word.txt", 'w')
        for w in list(self.powerful__dict.keys()):
            pw = (self.powerful__dict[w][1]+1)/(self.powerful__dict[w][0] + self.powerful__dict[w][1]+2)
            model_outf.write(w+" "+str(pw)+"\n")




if __name__ == '__main__':
    trainor = TrainStatisFeature()
    trainor.train_powerful_word()


