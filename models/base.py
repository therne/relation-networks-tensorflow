
import tensorflow as tf

class BaseModel:

    def __init__(self, data_info, params, training=True):
        self.data_info = data_info
        self.params = params
        self.training = training

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def loss(self, logits, answer):
        # Calculate cross-entropy losses
        with tf.variable_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
            losses = tf.reduce_mean(cross_entropy)
        return losses

    def accuracy(self, logits, answer):
        # Calculate accuracy
        with tf.name_scope('Accuracy'):
            corrects = tf.equal(tf.argmax(logits, axis=1), answer)
            num_corrects = tf.reduce_sum(tf.to_int32(corrects))

            tf.add_to_collection('num_examples', tf.shape(logits)[0])
            tf.add_to_collection('correct_examples', num_corrects)

            accuracy = tf.get_collection('correct_examples') / tf.get_collection('num_examples')

        return accuracy

    def train(self, loss, learning_rate, optimizer=tf.train.AdamOptimizer):
        train_op = optimizer(learning_rate).minimize(loss, global_step=self.global_step)
        return train_op




