source_file = "../data/atec_nlp_sim_train.csv"
user_dict = "../data/user_dict"
tf_out_dir = "../data/tfrecord/"
word_dict = "../data/word_dict.txt"

num_validation_sentences = 5000
train_output_shards = 10
validation_output_shards = 1

text_1_min=30
text_2_min=30

import tensorflow as tf
import jieba
import data_utils
import os
import random

jieba.load_userdict(user_dict)


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


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=[int(v) for v in value]))


def handle_one_example(words_1, text_1_len, words_2, text_2_len, label):

    example = tf.train.SequenceExample()

    w_1 = example.feature_lists.feature_list["text_1"]
    for w in words_1:
        w_1.feature.add().int64_list.value.append(w)

    w_2 = example.feature_lists.feature_list["text_2"]
    for l in words_2:
        w_2.feature.add().int64_list.value.append(l)
    example.feature_lists.feature_list["text_1_len"].feature.add().int64_list.value.append(text_1_len)
    example.feature_lists.feature_list["text_2_len"].feature.add().int64_list.value.append(text_2_len)

    y= [0] * 2
    y[int(label)] = 1
    f_y = example.feature_lists.feature_list['label']
    for l in y:
        f_y.feature.add().int64_list.value.append(l)

    return example.SerializeToString()



def create_one_example(word_to_id, line):
    line = line.strip("\n").split("\t")
    text_1 = line[1]
    words_1, text_1_len = format_sent(word_to_id, text_1, text_1_min)
    text_2 = line[2]
    words_2, text_2_len = format_sent(word_to_id, text_2, text_2_min)
    label = line[3]
    example_list = []
    # todo
    if(int(label) == 0):
        example_list.append(handle_one_example(words_1, text_1_len, words_2, text_2_len, label))
    else:
        for i in range(2):
            example_list.append(handle_one_example(words_1, text_1_len, words_2, text_2_len, label))
    # example_list.append(handle_one_example(words_1, text_1_len, words_2, text_2_len, label))
    return example_list



def create_tf_record(source_path, tf_out_dir):
    f = open(source_path)
    word_to_id, id_to_word = data_utils.load_word_dict(word_dict)

    if not  os.path.exists(tf_out_dir):
        os.mkdir(tf_out_dir)

    train_writer = tf.python_io.TFRecordWriter(os.path.join(tf_out_dir, "train.tfrecord"))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(tf_out_dir, "test.tfrecord"))
    all_example = []
    train_rate = 0.8
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print("create tf record : ", i)
        example_list = create_one_example(word_to_id, line)
        all_example.extend(example_list)
        # for example in example_list:
        #     writer.write(example)
    random.shuffle(all_example)
    all_count = len(all_example)
    print('all examle %d'%(all_count))
    for i, example in enumerate(all_example):
        if i < all_count * 0.9:
            train_writer.write(example)
        else:
            test_writer.write(example)




if __name__ == '__main__':
    create_tf_record(source_file, tf_out_dir)
