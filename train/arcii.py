
import tensorflow as tf


class ARCII(object):
    def __init__(self, config, embedding_array):

        self.learning_rate = config['learning_rate']
        self.decay_steps = config['decay_steps']
        self.decay_rate = config['decay_rate']
        self.optimizer = config['optimizer']
        self.clip_gradients = config['clip_gradients']

        self.max_x1_len = config['max_x1_len']
        self.max_x2_len = config['max_x2_len']

        self.kernal_size = config['kernal_size']
        self.kernal_count = config['kernal_count']

        self.x1_max_pool_size = config['x1_max_pool_size']
        self.x2_max_pool_size = config['x2_max_pool_size']
        self.embedding_size = config['embedding_size']

        self.keep_drop_out = tf.placeholder_with_default(1.0,[], name="keep_drop_out")

        # todo
        self.initializer = tf.random_normal_initializer(stddev = 0.1)

        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, None))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, None))

        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None,))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None,))

        self.batch_size = tf.shape(self.X1)[0]

        self.Y = tf.placeholder(tf.int32, name='Y', shape=(None, None))
        self.global_step = tf.Variable(0, trainable=False, name = "global_step")

        self.embedding = tf.get_variable(name='embedding', initializer=embedding_array, dtype=tf.float32, trainable=True)

        x1_conv = self.conv_laryer(self.X1, self.kernal_size, self.kernal_count, self.x1_max_pool_size, "x1", self.max_x1_len)
        x2_conv = self.conv_laryer(self.X2, self.kernal_size, self.kernal_count, self.x2_max_pool_size, "x2", self.max_x2_len)
        # match

        self.loss_layer(x1_conv, x2_conv)
        self.train_op = self.train()

    def conv_laryer_first(self, x, kernal_size, kernal_count, pool_size, prefix, text_len):
        # x: batch_size, sent_len
        # embed_x: batch_size, sent_len, embedding_size
        embed_x = tf.nn.embedding_lookup(self.embedding, x)
        print("embed_x ===>", embed_x)

        # batch_size, sent_len, embedding_size, 1
        embed_x_expand = tf.expand_dims(embed_x, -1)
        print("embed_x_expand ===>", embed_x_expand)
        # kernal_size , embedding_size, 1,
        # filter = tf.get_variable("%s_kernal"%(prefix, [kernal_size, self.embedding_size, self.kernal_count]), initializer=self.initializer)
        # conv = tf.nn.conv1d(embed_x,filters=filter, stride=[])

        filter = tf.get_variable("%s_kernal"%prefix, [kernal_size, self.embedding_size, 1, self.kernal_count],
                                     initializer = self.initializer)
        print("filter===> ", filter)

        #conv: batch_size, sent_len-filter_size + 1,  1 , filter_count
        conv = tf.nn.conv2d(embed_x_expand, filter, strides=[1, 1, 1, 1], padding="VALID", name = prefix + "_conv",)
        print("conv ===>", conv)
        b = tf.get_variable("%s_b" % prefix, [self.kernal_count])

        # h: batch_size, sent_len-filter_size +1, 1, filter_count
        h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

        # pooled: batch_size, (sent_len-filer_size + 1)/pool_size, 1, filer_count
        pooled = tf.nn.max_pool(h,
                                ksize = [1, text_len-kernal_size+1, 1, 1],
                                strides = [1, 1, 1, 1],
                                padding = 'VALID',
                                name = prefix + "_pool")
        # return : batch_size, (sent_len-filter_size+1)/pool_size,1, filter_count/2
        print("pooled ===>", pooled)
        return tf.reshape(pooled, (-1, kernal_count))

    def match(self, x1, x2):
        x = tf.stack(x1)

    def loss_layer(self, x1, x2):
        x = tf.concat([x1, x2], axis=1)
        print("contact ====>", x)
        x_droup = tf.nn.dropout(x, keep_prob=self.keep_drop_out)

        w = tf.get_variable(name="w", shape=(2*self.kernal_count,2))
        b = tf.get_variable(name='b', shape=(2))

        logits = tf.nn.xw_plus_b(x_droup, w, b, name="logits")
        print("logits ===>", logits)
        self.predict = tf.argmax(logits, axis=1, name="predict")
        self.prob = tf.nn.softmax(logits, name="prob")
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y )
        self.loss = tf.reduce_mean(loss, name="loss")

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step,
                                                   self.decay_steps,
                                                   self.decay_rate,
                                                   staircase=True)


        train_op = tf.contrib.layers.optimize_loss(self.loss,
                                                   global_step=self.global_step,
                                                   learning_rate=learning_rate,
                                                   optimizer=self.optimizer,
                                                   )
        return train_op