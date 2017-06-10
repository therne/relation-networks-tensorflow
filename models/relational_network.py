import collections

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import Conv2D, Activation, Dense, Dropout, BatchNormalization
from tensorflow.contrib.rnn import LSTMCell

from models.base import BaseModel
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
    'f_units',
    'dropout_rate',
    'answer_vocab_size',
])


class RelationNetwork(BaseModel):
    """ Relation Network (https://arxiv.org/abs/1706.01427) """

    def infer(self, image, question, seq_len):
        # image: [num_batch, 128, 128, 3]
        # question: [num_batch, max_seq_len]
        # seq_len: [num_batch]

        # Extract features
        feat_question = self.extract_question_feature(question, seq_len)
        feature_maps = self.extract_image_feature(image)

        # G network that tries to find every possible relations.
        g_outputs = []
        for i in range(25):
            object_i = tf.squeeze(feature_maps[:, i/5, i%5, :])  # [num_batch, units]
            for j in range(25):
                object_j = tf.squeeze(feature_maps[:, j/5, j%5, :])

                # feed object pair and question feature into the G network.
                g_input = tf.concat([object_i, object_j, feat_question], axis=1)
                g_outputs.append(self.g_network(g_input))

        # F network that combines all features.
        f_input = tf.add_n(g_outputs)
        logits = self.f_network(f_input)
        return logits

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
            x = Dense(units, activation='relu')(x)
            output = x

        return output

    def f_network(self, f_input):
        """ Combines all output features from G and provides result.
        :param f_input: Added outputs from G with shape [batch_size, f_units]
        :return: logits with shape [batch_size, answer_vocab_size]
        """
        vocab_size = self.data_info.answer_vocab_size
        dropout_rate = self.params.dropout_rate
        units = self.params.f_units

        with tf.variable_scope('F'):
            x = Dense(units, activation='relu')(f_input)
            x = Dropout(dropout_rate)(x, training=self.training)
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout_rate)(x, training=self.training)
            x = Dense(vocab_size, activation='relu')(x)
            output = x

        return output
