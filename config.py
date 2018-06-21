import os


class CONFIG(object):
    batch_size = 10
    n_epochs = 500
    train_space = 90
    max_steps = 140
    input_dim = 3
    output_dim = 3
    neurons_dim = 100
    learning_rate = 0.001
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
    UNI_GE_DATA_SETS = [1, 2, 5, 6, 7, 8]

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
