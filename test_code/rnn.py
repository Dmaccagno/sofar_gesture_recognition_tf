import tensorflow as tf
import numpy as np
import os
from test_code.file_utils import FilesUtil
from test_code.config import CONFIG
from sklearn.metrics import mean_squared_error

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


def loop_epochs(x, y, n_epochs, training_op, loss, batch_size, train_data_sets, validation_data_sets, seq_length,
                gesture_id,
                sess):
    validation_check = 0
    for epoch in range(n_epochs):
        for iteration in range(CONFIG.train_space // batch_size):
            bx, by, s = FilesUtil.feed_next_batch(train_data_sets, batch_size, CONFIG.max_steps)
            vx, vy, s = FilesUtil.feed_next_batch(validation_data_sets, batch_size, CONFIG.max_steps)
            sess.run(training_op, feed_dict={x: np.asarray(bx), y: np.asarray(by), seq_length: np.asarray(s)})
            mse_train = loss.eval(feed_dict={x: np.asarray(bx), y: np.asarray(by), seq_length: np.asarray(s)})
            mse_val = loss.eval(feed_dict={x: np.asarray(vx), y: np.asarray(vy), seq_length: np.asarray(s)})
            perc_diff = FilesUtil.get_percent_diff(mse_train, mse_val)
            print("EPOCH=", epoch, "MSE_train", mse_train, "MSE_validate", mse_val, "Percentage Diff",
                  perc_diff, "Gesture ", gesture_id)
            if (mse_train < 0.4 and mse_val < 0.4) and perc_diff > 10.0:
                validation_check += 1
            else:
                validation_check = 0
        if validation_check == 3:
            print("exiting...")
            return sess
            # print("EPOCH=", epoch, "MSE_validate", mse)
        if epoch % 100 == 0:
            print("epoch", epoch, "completed")
    return sess


def create_rnn_model(train_data_sets, validation_data_sets, gesture_id, batch_size, neurons_dim, n_epochs):
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
        sess = loop_epochs(x=x, y=y, n_epochs=n_epochs, training_op=training_op, loss=loss, batch_size=batch_size,
                           train_data_sets=train_data_sets, validation_data_sets=validation_data_sets,
                           seq_length=seq_length,
                           gesture_id=gesture_id, sess=sess)
        # validation_check = 0
        # for epoch in range(n_epochs):
        #     for iteration in range(CONFIG.train_space // batch_size):
        #         bx, by, s = FilesUtil.feed_next_batch(train_data_sets, batch_size, CONFIG.max_steps)
        #         vx, vy, s = FilesUtil.feed_next_batch(validation_data_sets, batch_size, CONFIG.max_steps)
        #         sess.run(training_op, feed_dict={x: np.asarray(bx), y: np.asarray(by), seq_length: np.asarray(s)})
        #         mse_train = loss.eval(feed_dict={x: np.asarray(bx), y: np.asarray(by), seq_length: np.asarray(s)})
        #         mse_val = loss.eval(feed_dict={x: np.asarray(vx), y: np.asarray(vy), seq_length: np.asarray(s)})
        #         perc_diff = FilesUtil.get_percent_diff(mse_train, mse_val)
        #         print("EPOCH=", epoch, "MSE_train", mse_train, "MSE_validate", mse_val, "Percentage Diff",
        #               perc_diff, "Gesture ", gesture_id)
        #         if mse_train < 1.0 and mse_val < 1.0 and perc_diff > 10.0:
        #             validation_check += 1
        #         elif validation_check == 3:
        #             print("exiting....")
        #             break
        #     if validation_check == 3:
        #         print("exiting..")
        #         break
        #         # print("EPOCH=", epoch, "MSE_validate", mse)
        #     if epoch % 100 == 0:
        #         print("epoch", epoch, "completed")
        model_path = os.path.join(os.getcwd(), 'model', 'model' + gesture_id, 'rnn_' + gesture_id + '.ckpt')
        saver.save(sess,
                   model_path)
