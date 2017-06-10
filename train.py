#!/usr/bin/env python3

import tensorflow as tf

from models.relational_network import DataInfo, HyperParams, RelationNetwork

flags = tf.app.flags

flags.DEFINE_float('lr', 2.5e-4, 'Learning rate.')
flags.DEFINE_integer('epoches', 500, 'Learning epoches.')
flags.DEFINE_string('data_path', 'data/clevr_data.tfrecords', 'Preprocessed data path. (tfrecords)')
flags.DEFINE_string('log_dir', 'run/logs/', 'Logging directory.')
flags.DEFINE_string('save_dir', 'run/checkpoints/', 'Model saving directory.')
flags.DEFINE_integer('save_interval', 15 * 60, 'Model save interval. (sec)')
flags.DEFINE_integer('summary_interval', 60, 'Summary saving interval. (sec)')

FLAGS = flags.FLAGS


def main(_):
    # Train supervisor that helps long-time training
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir,
                             save_model_secs=FLAGS.save_interval,
                             save_summaries_secs=FLAGS.summary_interval)

    with sv.managed_session() as sess:
        data_info = DataInfo()
        params = HyperParams()

        model = RelationNetwork(data_info, params)
        sess.run(tf.global_variables_initializer())

        # Read and feed to the graph
        image, question, seq_len, answer = read_data(FLAGS.data_path)
        logit_op = model.infer(image, question, seq_len)
        loss_op = model.loss(logit_op, answer)

        with tf.name_scope('Accuracy'):
            # Calculate accuracy
            corrects = tf.equal(tf.argmax(logit_op, axis=1), answer)
            accuracy = tf.reduce_sum(tf.to_int32(corrects))


        logits, losses, acc = sess.run([logit_op, loss_op, accuracy])

        placeholders = {
            model.image: None,
            model.question: None,
            model.answer: None
        }
        losses, logits = sess.run([model.losses, model.logits], placeholders)


def read_data(file_path):
    file_queue = tf.train.string_input_producer([file_path], num_epochs=FLAGS.epoches)
    reader = tf.TFRecordReader()

    # decode records.
    _, examples = reader.read(file_queue)
    example = tf.parse_single_example(examples, features={
        'image': tf.FixedLenFeature([], tf.string),
        'question': tf.VarLenFeature(tf.string),
        'answer': tf.VarLenFeature(tf.string),
    })

    image = tf.decode_raw(example['image'], tf.uint8)
    question = example['question']
    answer = example['answer']

    return image, question, None, answer


if __name__ == '__main__':
    tf.app.run()

