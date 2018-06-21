import tensorflow as tf
import numpy as np
import os
from file_utils import FilesUtil
from config import CONFIG
from sklearn.metrics import mean_squared_error
from math import sqrt

print(tf.__version__)


def get_mse(series1, series2):
    return mean_squared_error(series1, series2)


def evaluate_rnn_model(single_series, gesture_id, neurons_dim):
    tf.reset_default_graph()
    # todo put that parameter in configuration class
    # cut = CONFIG.max_steps
    x = tf.placeholder(tf.float32, [None, None, CONFIG.input_dim])
    # seq_length = tf.placeholder(tf.int32, [None])
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.LSTMCell(num_units=neurons_dim, activation=CONFIG.get_activation_function(gesture_id)),
        output_size=CONFIG.output_dim)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # x_new = FilesUtil.cut_and_reshape(single_series, cut)
        x_new = FilesUtil.reshape(single_series)
        model_path = os.path.join(os.getcwd(), 'model', 'model' + gesture_id, 'rnn_' + gesture_id + '.ckpt')
        saver.restore(sess, model_path)
        y_pre = sess.run(outputs, feed_dict={x: x_new})
    return x_new, y_pre


def create_rnn_model(train_data_sets, gesture_id, batch_size, neurons_dim, n_epochs):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, None, CONFIG.input_dim])
    y = tf.placeholder(tf.float32, [None, None, CONFIG.output_dim])
    seq_length = tf.placeholder(tf.int32, [None])

    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.LSTMCell(num_units=neurons_dim, activation=CONFIG.get_activation_function(gesture_id)),
        output_size=CONFIG.output_dim)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, sequence_length=seq_length)

    loss = tf.reduce_mean(tf.square(outputs - y))  # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=CONFIG.learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(CONFIG.train_space // batch_size):
                bx, by, s = FilesUtil.feed_next_batch(train_data_sets, batch_size, CONFIG.max_steps)
                sess.run(training_op, feed_dict={x: np.asarray(bx), y: np.asarray(by), seq_length: np.asarray(s)})
                if epoch % 100 == 0:
                    mse = loss.eval(feed_dict={x: bx, y: by, seq_length: np.asarray(s)})
                    print("EPOCH=", epoch, "MSE", mse)
        model_path = os.path.join(os.getcwd(), 'model', 'model' + gesture_id, 'rnn_' + gesture_id + '.ckpt')
        saver.save(sess,
                   model_path)
