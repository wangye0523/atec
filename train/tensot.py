#/usr/bin/env python
#-*-coding:utf-8-*-



import tensorflow as tf
import numpy as np


elems = tf.constant([[1,2,3],[2,4, 6]],dtype=tf.int64)
print(elems)
alternates = tf.map_fn(lambda x: (x, 2*x, -x), elems, dtype=(tf.int64, tf.int64, tf.int64))

with tf.Session() as sess:
    res = sess.run(alternates)
    for r in res:
        print("#"* 10)
        print(r)

    print(alternates)
