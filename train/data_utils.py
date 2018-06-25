


from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import numpy as np
import os

def load_word2vec(vec_path):
    model = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    return model

def load_word2vec_as_array(id_to_word, vec_path, dim_size):
    model = load_word2vec(vec_path)
    vocab_size = len(id_to_word)
    bound =  np.sqrt(6.0) / np.sqrt(vocab_size)
    embed = []
    for i in range(vocab_size):
        word = id_to_word[i]
        if word in model.vocab:
            vec = model.word_vec(word)
        else:
            vec = (np.random.uniform(-bound, bound, dim_size));
        embed.append(vec)
    # UNK
    vec = (np.random.uniform(-bound, bound, dim_size));
    embed.append(vec)
    return np.array(embed, dtype=np.float32)



def load_word_dict(word_dict_path):
    f = open(word_dict_path)
    word_dict = {}
    for i, line in enumerate(f):
        line = line.strip("\n").strip()
        line = line.split()
        w = line[0]
        w_id = int(line[1])
        word_dict[w] = w_id
    id_to_word = {v:k for k,v in word_dict.items()}
    return word_dict, id_to_word

class SegBatcher(object):
    def __init__(self, reader , record_file_dir, file_pattern, batch_size,  num_epochs=None):
        self._batch_size = batch_size
        self._epoch = 0
        self._step = 1.
        self.num_epochs = num_epochs
        self.reader = reader
        pattern = os.path.join(record_file_dir, file_pattern)
        # filenames = tf.matching_files([pattern])
        data_files = tf.gfile.Glob(pattern)
        print("found data: files:", data_files)
        self.next_batch_op = self.input_pipeline(data_files, self._batch_size, self.num_epochs)



    def example_parser(self, filename_queue):
        key, record_string = self.reader.read(filename_queue)

        features = {
            'text_1': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            'text_1_len': tf.FixedLenSequenceFeature([],dtype=tf.int64),
            'text_2': tf.FixedLenSequenceFeature([],dtype=tf.int64),
            'text_2_len': tf.FixedLenSequenceFeature([],dtype=tf.int64),
            'label': tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        _, example = tf.parse_single_sequence_example(serialized=record_string, sequence_features=features)


        text_1 = example['text_1']
        text_1_len = example['text_1_len']
        text_2  = example['text_2']
        text_2_len = example['text_2_len']
        y = example['label']
        return text_1, text_1_len, text_2, text_2_len, y

    def input_pipeline(self, filenames, batch_size,  num_epochs=None):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        text_1, text_1_len, text_2, text_2_len, y = self.example_parser(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        next_batch = tf.train.batch([text_1, text_1_len, text_2, text_2_len, y], batch_size=batch_size, capacity=capacity,
                                    dynamic_pad=True, allow_smaller_final_batch=True)

        return next_batch



if __name__ == '__main__':

    # sess = tf.Session()
    # sv = tf.train.Supervisor(logdir="./", save_model_secs=0, save_summaries_secs=0)
    with tf.Graph().as_default():
        with tf.train.MonitoredSession() as sess:
            reader = tf.TFRecordReader()
            batcher = SegBatcher(reader,"/home/rocky/dl/atec-nlp-sim/data/tfrecord", "train-?????-of-00010" , 32, 1)
            print(sess.should_stop())
            # while not sess.should_stop():
            #     sess.run(tf.global_variables_initializer())
            # sess.run(tf.initialize_all_variables())
            threads = tf.train.start_queue_runners(sess=sess)
            # tf.local_variables_initializer()
            batch = sess.run(batcher.next_batch_op)
            print(batch)
