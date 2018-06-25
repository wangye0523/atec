#!/usr/bin/env python
#-*-coding:utf-8-*-


import jieba

import tensorflow as tf
import numpy as np
import sys
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

jieba.load_userdict(os.path.join(CURRENT_PATH, "user_dict"))
word_dict_path = (os.path.join(CURRENT_PATH,"word_dict.txt"))

max_len = 30

def load_word_dict(word_dict_path):
    f = open(word_dict_path)
    word_dict = {}
    for i, line in enumerate(f):
        line = line.strip("\n").strip()
        line = line.split()
        w = line[0]
        w_id = int(line[1])
        word_dict[w] = w_id
    id_to_word = {v: k for k, v in word_dict.items()}
    return word_dict, id_to_word


word_dict, id_to_word = load_word_dict(word_dict_path)


def seg_sent(text):
    words = list(jieba.cut(text))
    return words


def format_sent(word_to_id, text, min_len):
    # todo UNK WORD
    N = len(word_to_id)
    words = seg_sent(text)
    words = [w for w in words if w]
    words = [word_to_id.get(w, len(word_to_id)) for w in words]
    text_len = len(words)

    if text_len < min_len:
        padding = [N] * (min_len - text_len)
        words.extend(padding)
    else:
        words = words[:min_len]
    return words, text_len


def load_graph(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def list_to_array(data_list, dtype=np.int32):
    array = np.array(data_list, dtype).reshape(1, len(data_list))
    return array


class Predictor(object):
    # cnn  fix_length 需要需要等于True


    def __init__(self, model_file):
        self.graph = load_graph(model_file)

        self.X1 = self.graph.get_operation_by_name("prefix/X1").outputs[0]
        self.X2 = self.graph.get_operation_by_name("prefix/X2").outputs[0]
        # self.lstm_keep_drop_out = self.graph.get_operation_by_name("prefix/lstm_keep_drop_out").outputs[0]
        # self.mlp_keep_drop_out = self.graph.get_operation_by_name("prefix/mlp_keep_drop_out").outputs[0]
        self.keep_drop_out = self.graph.get_operation_by_name("prefix/keep_drop_out").outputs[0]
        self.pred = self.graph.get_operation_by_name("prefix/predict").outputs[0]
        self.prob = self.graph.get_operation_by_name("prefix/prob").outputs[0]
        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()

    def predict(self, text_1, text_2):
        X1, X_1_len = format_sent(word_dict, text_1, 30)
        X2, X_2_len = format_sent(word_dict, text_2, 30)

        X1 = np.array([X1])
        X2 = np.array([X2])

        feed_dict = {
            self.X1: X1,
            self.X2: X2,
            # self.lstm_keep_drop_out:1.0,
            # self.mlp_keep_drop_out:1.0
            self.keep_drop_out:1.0
        }

        pred, prob = self.sess.run([self.pred, self.prob], feed_dict=feed_dict)
        return pred[0], prob


def process(inpath, outpath):
    predictor = Predictor(os.path.join(CURRENT_PATH, "model.pb"))

    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split("\t")
            pred_label, prob = predictor.predict(sen1, sen2)
            fout.write(lineno + '\t%d\n' % (pred_label))


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
