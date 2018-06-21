import os
import tensorflow as tf


class CONFIG(object):
    # special config for model 1
    n_epochs_1 = 500
    batch_size_1 = 10
    neurons_dim_1 = 20
    learning_rate_1 = 0.0001
    max_steps_1 = 150
    # special config for model 2
    n_epochs_2 = 500
    batch_size_2 = 10
    neurons_dim_2 = 20
    learning_rate_2 = 0.0001
    max_steps_2 = 150
    # special config for model 5
    n_epochs_5 = 1000
    batch_size_5 = 20
    neurons_dim_5 = 100
    learning_rate_5 = 0.001
    max_steps_5 = 150
    # special config for model 6
    n_epochs_6 = 500
    batch_size_6 = 10
    neurons_dim_6 = 20
    learning_rate_6 = 0.0001
    max_steps_6 = 150
    # special config for model 7
    n_epochs_7 = 500
    batch_size_7 = 10
    neurons_dim_7 = 20
    learning_rate_7 = 0.0001
    max_steps_7 = 150
    # special config for model 8
    n_epochs_8 = 500
    batch_size_8 = 10
    neurons_dim_8 = 20
    learning_rate_8 = 0.0001
    max_steps_8 = 150

    train_space = 45
    input_dim = 3
    output_dim = 3
    learning_rate = 0.001
    max_steps = 150
    G1_PATH = os.path.join(os.getcwd(), 'unigedataset/G1/')
    G2_PATH = os.path.join(os.getcwd(), 'unigedataset/G2/')
    G5_PATH = os.path.join(os.getcwd(), 'unigedataset/G5/')
    G6_PATH = os.path.join(os.getcwd(), 'unigedataset/G6/')
    G7_PATH = os.path.join(os.getcwd(), 'unigedataset/G7/')
    G8_PATH = os.path.join(os.getcwd(), 'unigedataset/G8/')
    col_id_1 = 0
    col_id_2 = 1
    col_id_3 = 2
    UNI_GE_PATH = os.path.join(os.getcwd(), 'unigedataset/')
    # UNI_GE_DATA_SETS = [1]
    UNI_GE_DATA_SETS = [1, 2, 5, 6, 7, 8]
    evaluation_runs = 10

    @staticmethod
    def get_path(gesture_id):
        if gesture_id == 1:
            return CONFIG.G1_PATH
        if gesture_id == 2:
            return CONFIG.G2_PATH
        if gesture_id == 5:
            return CONFIG.G5_PATH
        if gesture_id == 6:
            return CONFIG.G6_PATH
        if gesture_id == 7:
            return CONFIG.G7_PATH
        if gesture_id == 8:
            return CONFIG.G8_PATH

    @staticmethod
    def get_neurons_dim(gesture_id):
        if gesture_id == 1:
            return CONFIG.neurons_dim_1
        if gesture_id == 2:
            return CONFIG.neurons_dim_2
        if gesture_id == 5:
            return CONFIG.neurons_dim_5
        if gesture_id == 6:
            return CONFIG.neurons_dim_6
        if gesture_id == 7:
            return CONFIG.neurons_dim_7
        if gesture_id == 8:
            return CONFIG.neurons_dim_8

    @staticmethod
    def get_activation_function(gesture_id):
        if gesture_id == 1:
            return tf.nn.sigmoid
        if gesture_id == 2:
            return tf.nn.sigmoid
        if gesture_id == 5:
            return tf.nn.sigmoid
        if gesture_id == 6:
            return tf.nn.sigmoid
        if gesture_id == 7:
            return tf.nn.sigmoid
        if gesture_id == 8:
            return tf.nn.sigmoid
