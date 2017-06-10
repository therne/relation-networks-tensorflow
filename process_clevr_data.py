#!/usr/bin/env python3
# Preprocesses CLEVR data and converts into TFRecords format.

import tensorflow as tf
import json
import os

flags = tf.app.flags
flags.DEFINE_string('path', 'data/', 'The path of CLEVR dataset.')
flags.DEFINE_string('type', 'val', 'Type of dataset. [test|val|train]')
flags.DEFINE_integer('num_threads', 8, 'The number of image processing threads.')
flags.DEFINE_string('out_path', 'data/clevr_data.tfrecords', 'Output path of the preprocessed tfrecords file.')
FLAGS = flags.FLAGS

# wrapper functions for conveniences
_bytes_feature = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def make_example(image, question, answer):
    features = tf.train.Features(feature={
        'image': _bytes_feature(image),
        'question': _bytes_feature(question),
        'answer': _bytes_feature(answer)
    })
    return tf.train.Example(features=features)


def image_process_queue(path):
    # Load from filename queue
    filename_queue = tf.train.string_input_producer([path])
    filename = filename_queue.dequeue()

    # Decode image
    image_raw = tf.read_file(filename)
    image = tf.image.decode_png(image_raw)

    # Preprocessing - downsampling
    image = tf.image.resize_images(image, [128, 128])

    # Create image processing queue
    queue = tf.FIFOQueue(256, [tf.uint8])
    enqueue_op = queue.enqueue(image)
    queue_runner = tf.train.QueueRunner(queue, [enqueue_op] * FLAGS.num_threads)

    # Register to the procedure
    tf.train.add_queue_runner(queue_runner)

    return queue.dequeue()


def group_questions_by_image(questions):
    # Multiple questions are given at a image, so we need to group them by image.
    grouped_list = []
    current_group = []
    current_image = 0
    for question in questions:
        if question['image_index'] == current_image:
            needed_info = {key: question[key] for key in ('question', 'answer')}
            current_group.append(needed_info)
        else:
            grouped_list.append(current_group)
            current_image = question['image_index']
            current_group = []

    return grouped_list


def main(_):
    question_path = os.path.join(FLAGS.path, 'questions/CLEVR_{}_questions.json'.format(FLAGS.type))
    image_path = os.path.join(FLAGS.path, 'images', FLAGS.type)

    print("Reading questions...")
    with open(question_path) as file:
        questions = json.loads(file.read())['questions']

    # group questions by image index
    question_table = group_questions_by_image(questions)

    print("Done. Preprocessing images...")

    image_op = image_process_queue(image_path)
    writer = tf.python_io.TFRecordWriter(FLAGS.out_path)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)

        image_index = 0
        while not coord.should_stop():
            image = image_op.eval()

            # make multiple question examples per image.
            for question in question_table[image_index]:
                example = make_example(image, question['question'], question['answer'])
                writer.write(example.SerializeToString())

            image_index += 1

    print("Done. Preprocessed data is written on {}".format(FLAGS.out_path))


if __name__ == '__main__':
    tf.app.run()
