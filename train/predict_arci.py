


import jieba

import tensorflow as tf
import numpy as np
import json
import data_utils

jieba.load_userdict("../data/user_dict")
word_dict_path = "../data/word_dict.txt"
word_dict, id_to_word = data_utils.load_word_dict(word_dict_path)

max_len = 30

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
            self.keep_drop_out:1.0
        }

        pred, prob = self.sess.run([self.pred, self.prob], feed_dict=feed_dict)
        return pred[0], prob


if __name__ == '__main__':
    text_1 = "怎么更改花呗手机号码"
    text_2 = "我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号"
    predictor = Predictor("../results/arci.pb", )
    f = open("../data/atec_nlp_sim_train.csv")
    true_list = []
    pred_list = []
    from sklearn.metrics.classification import classification_report
    for i, line in enumerate(f):
        if i > 1000:
            break
        line = line.strip("\n").split("\t")
        label = line[-1]
        text_1 = line[1]
        text_2 = line[2]
        pred_label, prob = predictor.predict(text_1, text_2)
        true_list.append(int(label))
        pred_list.append(pred_label)
        print(i, label, pred_label, text_1 +"|" +  text_2, prob[0])
    print("result: ")
    print(classification_report(true_list, pred_list, ))
