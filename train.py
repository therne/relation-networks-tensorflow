#!/usr/bin/env python3

import tensorflow as tf
from model import DataInfo, HyperParams, RelationalNetwork

flags = tf.app.flags

flags.DEFINE_float('lr', 2.5e-4, 'Learning rate.')
flags.DEFINE_integer('embed_size', 384, 'Word embedding size.')
flags.DEFINE_string('log_dir', 'run/logs/', 'Logging directory.')
flags.DEFINE_string('save_dir', 'run/checkpoints/', 'Model saving directory.')
flags.DEFINE_integer('save_interval', 15 * 60, 'Model save interval. (sec)')
flags.DEFINE_integer('summary_interval', 60, 'Summary saving interval. (sec)')

FLAGS = flags.FLAGS


def main(_):
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir,
                             save_model_secs=FLAGS.save_interval,
                             save_summaries_secs=FLAGS.summary_interval)
    config = tf.ConfigProto(allow_soft_placement=True)

    with sv.managed_session(config=config) as sess:
        data_info = DataInfo()
        params = HyperParams()
        model = RelationalNetwork(data_info, params)

        placeholders = {
            model.image: None,
            model.question: None,
            model.answer: None
        }
        losses, logits = sess.run([model.losses, model.logits], placeholders)



if __name__ == '__main__':
    tf.app.run()
