import collections

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import Conv2D, Activation, Dense, Dropout, BatchNormalization
from tensorflow.contrib.rnn import LSTMCell

from utils import weight

# Data information
DataInfo = collections.namedtuple('DataInfo', [
    'img_size',
    'vocab_size',
    'answer_vocab_size',
    'max_sent_length',
])

# Hyperparameters
HyperParams = collections.namedtuple('HyperParams', [
    'embed_dims',
    'g_units',
    'g_depths',
    'dropout_rate',
    'answer_vocab_size',
])


class RelationalNetwork:
    """ Relational Network (https://arxiv.org/abs/1706.01427) """

    def __init__(self, data_info: DataInfo, params: HyperParams, training=True):
        self.data_info = data_info
        self.params = params
        self.training = training

        # Inputs
        image = tf.placeholder(tf.float32, shape=[None, data_info.img_size, data_info.img_size, 3], name='image')
        question = tf.placeholder(tf.int32, shape=[None, data_info.max_sent_length], name='question')
        answer = tf.placeholder(tf.int32, shape=[None], name='answer')
        seq_len = tf.placeholder(tf.int32, shape=[None], name='question_len')

        # Extract features
        feat_question = self.extract_question_feature(question, seq_len)
        feature_maps = self.extract_image_feature(image)

        # G network that tries to find every possible relations.
        g_outputs = []
        for i in range(25):
            object_i = feature_maps[:, :, i/5, i%5]
            for j in range(25):
                object_j = feature_maps[:, :, j/5, j%5]

                # feed object pair and question feature into the G network.
                g_input = tf.concat([object_i, object_j, feat_question], axis=1)
                g_outputs.append(self.g_network(g_input))

        # F network that combines all features.
        f_input = tf.add_n(g_outputs)
        logits = self.f_network(f_input)

        # Calculate cross-entropy losses
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
        losses = tf.reduce_mean(cross_entropy)

        self.image = image
        self.question = question
        self.answer = answer
        self.seq_len = seq_len

        self.losses = losses
        self.logits = logits

    def extract_question_feature(self, question, seq_len):
        """ Extracts question feature using LSTM.
        :param question A Tensor with shape []
        :param seq_len A Tensor containing length of a quesiton - with shape []
        """
        vocab_size = self.data_info.vocab_size
        embed_dims = self.params.embed_dims

        with tf.variable_scope('Embedding'):
            embedding = weight('embedding', [vocab_size, embed_dims])
            question_embed = tf.nn.embedding_lookup(embedding, question)  # [embed_dims]

        with tf.variable_scope('QuestionLSTM'):
            lstm_cell = LSTMCell(128)
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, question_embed, seq_len, dtype=tf.float32)

        return state

    def extract_image_feature(self, image):
        """ Extracts image feature map using ConvNet.
            :param image A Tensor with shape [batch_size, img_size, img_size, 3]
        """
        with tf.variable_scope('ImageConv'):
            x = Conv2D(24, (3, 3), strides=(2, 2))(image)
            x = BatchNormalization()(x, training=self.training)
            x = Activation('relu')(x)

            x = Conv2D(24, (3, 3), strides=(2, 2))(x)
            x = BatchNormalization()(x, training=self.training)
            x = Activation('relu')(x)

            x = Conv2D(24, (3, 3), strides=(2, 2))(x)
            x = BatchNormalization()(x, training=self.training)
            output = Activation('relu')(x)

        return output

    def g_network(self, objects):
        """ Infers the ways in which two objects are related. 
        :param: objects: A concatenated tensor of two objects and question features - with shape []
        :return: A "relation" with shape [batch_size, g_units]
        """
        units = self.params.g_units
        with tf.variable_scope('G', reuse=True):
            x = Dense(units, activation='relu')(objects)
            x = Dense(units, activation='relu')(x)
            x = Dense(units, activation='relu')(x)
            output = Dense(units, activation='relu')(x)

        return output

    def f_network(self, f_input):
        """ Combines all output features from G and provides result.
        :param f_input: Added outputs from G with shape [batch_size, 256]
        :return: logits with shape [batch_size, answer]
        """
        dropout_rate = self.params.dropout_rate
        vocab_size = self.data_info.answer_vocab_size

        with tf.variable_scope('F'):
            x = Dense(256, activation='relu')(f_input)
            x = Dropout(dropout_rate)(x, training=self.training)
            x = Dense(256, activation='relu')(x)
            x = Dropout(dropout_rate)(x, training=self.training)
            output = Dense(vocab_size, activation='relu')(x)

        return output

