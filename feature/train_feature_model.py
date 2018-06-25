#/usr/bin/env python
#-*-coding:utf-8-*-


SOURCE_PATH = "../data/atec_nlp_sim_train.csv"
from papre_process import PapreProcessor
from gensim import corpora
from gensim.models import TfidfModel, LsiModel, Word2Vec

class TrainModel(object):
    def __init__(self):
        self.papre_processor = PapreProcessor("../data/user_dict", "../data/stopwords.txt")

    def train(self):
        clean_text_list = []
        print("start load and clean data ...")
        with open(SOURCE_PATH) as f:
            for _, line in enumerate(f):
                line= line.strip("\n").strip()
                line = line.split("\t")
                text_1 = line[1]
                text_2 = line[2]
                pre_1 = self.papre_processor.process(text_1)
                pre_2 = self.papre_processor.process(text_2)
                clean_text_list.append(pre_1.clean_words)
                clean_text_list.append(pre_2.clean_words)

        print("create corpus....")
        dictionary = corpora.Dictionary(clean_text_list)
        dictionary.save("../models/gensim.dict")
        corpus = [dictionary.doc2bow(text) for text in clean_text_list]
        corpora.MmCorpus.serialize('../models/gensim.mm', corpus)

        print("train tfidf ....")
        tfidf = TfidfModel(corpus, normalize=True)
        tfidf.save("../models/gensim.tfidf")
        corpus_tfidf = tfidf[corpus]
        print("train lsi.... ")
        lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
        lsi.print_topics(10)
        corpus_lsi = lsi[corpus_tfidf]
        for doc in corpus_lsi[:10]:
            print(doc)
        lsi.save('../models/gensim.lsi')

        print("train word2vec ....")
        word2vec = Word2Vec(clean_text_list, size=200, window=8, min_count=2, workers=4)
        word2vec.save("../models/gensim.word2vec")

if __name__ == '__main__':
    train_model = TrainModel()
    train_model.train()