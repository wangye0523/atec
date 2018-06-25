

import mvlstm

import json
import tensorflow as tf
import data_utils
import numpy as np
from sklearn.metrics.classification import classification_report
from sklearn.metrics.classification import accuracy_score

config_path = "./mvlstm.json"
train_tf_record = "../data/tfrecord"
train_pattern= "train.tfrecord"
test_pattern = "test.tfrecord"
out_dir = "../results/mvlstm"

batch_size= 1024
num_epochs = 2

def load_config(config_path):
    config = json.loads(open(config_path).read())
    return config

def main(argv):
    config = load_config(config_path)
    word_dict,id_to_word = data_utils.load_word_dict(config['word_dict'])
    embedding_array = data_utils.load_word2vec_as_array(id_to_word, config['embedding_path'], config["word_dim_size"])
    model_config = config.get("model_config")
    print(model_config)

    iteration = 0
    with tf.Graph().as_default():
        model = mvlstm.MVLSTM(model_config, embedding_array)
        reader = tf.TFRecordReader()
        batcher = data_utils.SegBatcher(reader,train_tf_record, train_pattern , batch_size, num_epochs)
        dev_batcher = data_utils.SegBatcher(reader, train_tf_record, test_pattern, batch_size, num_epochs)
        saver = tf.train.Saver()
        checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir=out_dir,
                                                       checkpoint_basename="mvlstm",
                                                       save_steps=10
                                                       )
        with tf.train.MonitoredSession(hooks=[checkpoint_hook]) as sess:
            print(sess.should_stop())
            threads = tf.train.start_queue_runners(sess=sess)
            dev_list  = []
            try:
                dev_batch = sess.run(dev_batcher.next_batch_op)
                text_1, text_1_len, text_2, text_2_len,y = dev_batch

                text_1_len = text_1_len.reshape(-1)
                text_2_len = text_2_len.reshape(-1)
                dev_list.append([text_1, text_1_len, text_2, text_2_len, y])

            except Exception as e:
                print(e)
            while not sess.should_stop():
                batch = sess.run(batcher.next_batch_op)
                text_1, text_1_len, text_2, text_2_len,y = batch
                text_1_len = text_1_len.reshape(-1)
                text_2_len = text_2_len.reshape(-1)
                feed_dict = {
                    model.X1: text_1,
                    model.X1_len: text_1_len,
                    model.X2: text_2,
                    model.X2_len: text_2_len,
                    model.Y: y,
                    model.lstm_keep_drop_out : 0.5,
                    model.mlp_keep_drop_out:0.75
                }
                loss, global_step, _ = sess.run([model.loss, model.global_step, model.train_op], feed_dict=feed_dict)
                true_list = []
                predict_list = []

                print("%d iteration train loss: %f"%(global_step, loss))
                if global_step % 10 ==0:
                    # todo test
                    for text_1, text_1_len, text_2, text_2_len, y in dev_list:
                        feed_dict = {
                            model.X1: text_1,
                            model.X1_len: text_1_len,
                            model.X2: text_2,
                            model.X2_len: text_2_len,
                            model.Y: y,
                            model.lstm_keep_drop_out : 1.0,
                            model.mlp_keep_drop_out: 1.0
                        }
                        predict, prob = sess.run([model.predict, model.prob],feed_dict=feed_dict)
                        t = np.argmax(y, axis=1)
                        true_list.extend(t)
                        predict_list.extend(predict)
                    acc = accuracy_score(true_list,predict_list)
                    print("%d dev acc: %f"%(global_step, acc))
                    print(classification_report(true_list, predict_list))


if __name__ == '__main__':
    tf.app.run()
