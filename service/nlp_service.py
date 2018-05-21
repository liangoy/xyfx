import tensorflow as tf
import jieba
import pymongo
import numpy as np
from util import ml
import json
import jieba


class NlpService():
    word_size = 100
    embedding_size = 2
    batch_size = 8192
    db = pymongo.MongoClient().xingqiao
    w2i = {i['word']: i['index'] for i in db.w2iGtTfidfGt5000.find()}
    fill1=[[1]*word_size]*batch_size

    def __init__(self):

        # ..........................................................................
        print('data pre-processing is done')

        self.x = tf.placeholder(shape=[self.batch_size, self.word_size], dtype=tf.int32)
        self.y_ = tf.placeholder(shape=[self.batch_size], dtype=tf.float32)

        embeddings = tf.Variable(
            tf.random_uniform([566 + 2, self.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.x)
        X = tf.reshape(embed, [self.batch_size, self.word_size, self.embedding_size, 1])

        c1 = ml.conv2d(X, conv_filter=[10, self.embedding_size, 1, 2], padding='VALID', ksize=[1, 10, 1, 1],
                       pool_stride=[1, 4, 1, 1],
                       pool_padding='SAME')
        c2 = ml.conv2d(c1, conv_filter=[4, 1, 2, 4], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 5, 1, 1],
                       pool_padding='SAME')
        c3 = ml.conv2d(c2, conv_filter=[5, 1, 4, 8], padding='VALID', ksize=[1, 1, 1, 1], pool_stride=[1, 1, 1, 1],
                       pool_padding='VALID')

        out = tf.reshape(c3, shape=[self.batch_size, 8])
        self.y = tf.nn.sigmoid(ml.layer_basic(out, 1))[:, 0]
        # ...................................................................
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, '/home/liangoy/Desktop/project/xingqiao_model/msgTfidf566')

    def cut_word(self, text, size=None):
        if size:
            return ([self.w2i.get(i, 1) for i in jieba.cut(text)] + [0] * size)[:size]
        else:
            return [self.w2i.get(i, 1) for i in jieba.cut(text)]

    def get_score(self, lis, m='r'):
        cnt = len(lis)
        lis = lis + self.fill1[:self.batch_size - cnt % self.batch_size]
        score = []
        for i in range(cnt // self.batch_size + 1):
            d = lis[i * self.batch_size:i * self.batch_size + self.batch_size]
            score.extend(list(self.sess.run(self.y, feed_dict={self.x: d})))
        return np.mean(score[:cnt])


nlp_service = NlpService()
