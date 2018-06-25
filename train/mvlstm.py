

import tensorflow as tf
from tensorflow.contrib import rnn

class MVLSTM(object):
    def __init__(self, config, embedding_array):
        self.c = config['c']
        self.hidden_size = config['hidden_size']
        self.max_x1_len = config['max_x1_len']
        self.max_x2_len = config['max_x2_len']
        self.k_max = config['max_pool_top_k']

        self.learning_rate = config['learning_rate']
        self.decay_steps = config['decay_steps']
        self.decay_rate = config['decay_rate']
        self.optimizer = config['optimizer']
        self.clip_gradients = config['clip_gradients']

        self.global_step = tf.Variable(0, trainable=False, name = "global_step")


        # batch_size * batch_max_len
        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, None))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, None))

        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None, ))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None, ))
        self.batch_size = tf.shape(self.X1)[0]

        self.lstm_keep_drop_out = tf.placeholder_with_default(1.0, [], name="lstm_keep_drop_out")
        self.mlp_keep_drop_out = tf.placeholder_with_default(1.0, [],name="mlp_keep_drop_out")

        self.Y = tf.placeholder(tf.int32, name='Y', shape=(None, None))

        self.embedding = tf.get_variable(name='embedding', initializer = embedding_array, dtype=tf.float32, trainable=True)
        x1_rnn_output = self.bi_lstm_layer(self.X1, "x_1")
        x2_rnn_output =  self.bi_lstm_layer(self.X2, "x_2")
        # self.dot_layer(x1_rnn_output, x2_rnn_output)
        self.tensor_layer(x1_rnn_output, x2_rnn_output)
        self.loss()
        self.train_op = self.train()



    def bi_lstm_layer(self, x, name="x_1"):
        print("x====>", x)
        embedded_words = tf.nn.embedding_lookup(self.embedding, ids=x, name="emb"+name)

        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size, name="fw_"+name)
        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.lstm_keep_drop_out)

        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size, name="bw_"+name)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.lstm_keep_drop_out)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words, dtype=tf.float32)
        # batch_size,  max_len, hidden_size * 2

        output_rnn = tf.concat(outputs, axis=2, name="rnn_out_"+name)
        print("output_rnn ===>", output_rnn)
        return output_rnn


    def dot_layer(self, x_1, x_2):
        # x1: batch_size, x_1_len, hidden_size   x2:batch_size, x_2_len, hidden_size
        self.match = tf.einsum('abd,acd->abc', x_1, x_2)
        ### batch_size, max_x1_len, max_x2_len
        self.cross = tf.expand_dims(self.match , 3)
        print("cross === >", self.cross)
        cross_reshape = tf.reshape(self.cross, (self.batch_size, self.max_x1_len * self.max_x2_len))
        # k_max_pool: batch_size, self.k_max
        print("k_max_pool ===> ", self.k_max_pool)
        self.k_max_pool = tf.nn.top_k(cross_reshape, k=self.k_max, name="k_max_pool")[0]



    def loss(self):
        self.w_r = tf.get_variable(name="w_r", shape=(self.k_max * self.c, 2), dtype=tf.float32)
        self.b_r = tf.get_variable(name="b_r", shape=(2), dtype=tf.float32)
        # todo drop out
        dropout = tf.nn.dropout(self.k_max_pool, keep_prob=self.mlp_keep_drop_out)
        logits = tf.matmul(dropout, self.w_r) + self.b_r
        self.predict = tf.argmax(logits, axis=1, name="predict")
        self.prob = tf.nn.softmax(logits, name="prob")
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels= self.Y)
        # l2 = tf.nn.l2_loss(self.w_r)
        loss = tf.reduce_mean(loss)
        self.loss = loss



    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step,
                                                   self.decay_steps,
                                                   self.decay_rate,
                                                   staircase = True)
        #
        # train_op = tf.contrib.layers.optimize_loss(self.loss,
        #                                            global_step = self.global_step,
        #                                            learning_rate = learning_rate,
        #                                            optimizer = self.optimizer,
        #                                            clip_gradients = self.clip_gradients)

        train_op = tf.contrib.layers.optimize_loss(self.loss,
                                                   global_step = self.global_step,
                                                   learning_rate = learning_rate,
                                                   optimizer = self.optimizer,
                                                   )
        return train_op



    def tensor_layer(self, x_1, x_2):
        # x_1 : batch_size , x1_max_len, hidden_size *2
        # x_2 : batch_size , x2_max_len, hidden_size *2

        # M : 2* hidden_size, 2*hidden_size
        self.M = tf.get_variable(name="M", shape=(2 * self.hidden_size, 2* self.hidden_size, self.c), dtype=tf.float32)
        # corss: batch_size, c, x1_max_len, x2_max_len,

        cross_a = tf.einsum("abh,hdc->abdc", x_1, self.M)
        self.cross = tf.einsum("afh,abhc->acbf",x_2, cross_a)
        print("crosss ===>", self.cross)
        cross_reshape = tf.reshape(self.cross, shape=(self.batch_size, self.c, -1))
        # batch_size, c, k
        k_max_pool = tf.nn.top_k(cross_reshape, k=self.k_max)[0]
        self.k_max_pool = tf.reshape(k_max_pool, shape=(self.batch_size, -1))
        print("k_max_pool ===> ",self.k_max_pool)

