from rnn import create_rnn_model, evaluate_rnn_model, get_mse
from file_utils import FilesUtil
from config import CONFIG
from pandas import DataFrame
from collections import defaultdict
import numpy as np
import os
import itertools
from collections import Counter


class Launcher(object):

    @staticmethod
    def launch_evaluation():
        confusion_matrix = np.empty([6, 6])
        print("-------------------------- gesture 1 --------------------------")
        result = Launcher.evaluate_model(1)
        g1_list = list()
        for i in range(CONFIG.evaluation_runs):
            g1_list.append(result.loc[result[i] == result[i].min()].index[0])

        index = 0
        for gestures in enumerate(CONFIG.UNI_GE_DATA_SETS):
            if gestures[1] in dict(Counter(g1_list)):
                confusion_matrix[0][index] = dict(Counter(g1_list))[gestures[1]]
            else:
                confusion_matrix[0][index] = 0
            index += 1

        print("-------------------------- gesture 2 --------------------------")
        g2_list = list()
        result = Launcher.evaluate_model(2)
        for i in range(CONFIG.evaluation_runs):
            g2_list.append(result.loc[result[i] == result[i].min()].index[0])

        index = 0
        for gestures in enumerate(CONFIG.UNI_GE_DATA_SETS):
            if gestures[1] in dict(Counter(g2_list)):
                confusion_matrix[1][index] = dict(Counter(g2_list))[gestures[1]]
            else:
                confusion_matrix[1][index] = 0
            index += 1

        print("-------------------------- gesture 5 --------------------------")
        g5_list = list()
        result = Launcher.evaluate_model(5)
        for i in range(CONFIG.evaluation_runs):
            g5_list.append(result.loc[result[i] == result[i].min()].index[0])

        index = 0
        for gestures in enumerate(CONFIG.UNI_GE_DATA_SETS):
            if gestures[1] in dict(Counter(g5_list)):
                confusion_matrix[2][index] = dict(Counter(g5_list))[gestures[1]]
            else:
                confusion_matrix[2][index] = 0
            index += 1

        print("-------------------------- gesture 6 --------------------------")
        g6_list = list()
        result = Launcher.evaluate_model(6)
        for i in range(CONFIG.evaluation_runs):
            g6_list.append(result.loc[result[i] == result[i].min()].index[0])

        index = 0
        for gestures in enumerate(CONFIG.UNI_GE_DATA_SETS):
            if gestures[1] in dict(Counter(g6_list)):
                confusion_matrix[3][index] = dict(Counter(g6_list))[gestures[1]]
            else:
                confusion_matrix[3][index] = 0
            index += 1

        print("-------------------------- gesture 7 --------------------------")
        g7_list = list()
        result = Launcher.evaluate_model(7)
        for i in range(CONFIG.evaluation_runs):
            g7_list.append(result.loc[result[i] == result[i].min()].index[0])

        index = 0
        for gestures in enumerate(CONFIG.UNI_GE_DATA_SETS):
            if gestures[1] in dict(Counter(g7_list)):
                confusion_matrix[4][index] = dict(Counter(g7_list))[gestures[1]]
            else:
                confusion_matrix[4][index] = 0
            index += 1

        print("-------------------------- gesture 8 --------------------------")
        g8_list = list()
        result = Launcher.evaluate_model(8)
        for i in range(CONFIG.evaluation_runs):
            g8_list.append(result.loc[result[i] == result[i].min()].index[0])

        index = 0
        for gestures in enumerate(CONFIG.UNI_GE_DATA_SETS):
            if gestures[1] in dict(Counter(g8_list)):
                confusion_matrix[5][index] = dict(Counter(g8_list))[gestures[1]]
            else:
                confusion_matrix[5][index] = 0
            index += 1

        print(confusion_matrix)

    @staticmethod
    def train_models():

        # train_sets1, test_sets1 = FilesUtil.split_data_set(CONFIG.G1_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets1, gesture_id=str(1), batch_size=CONFIG.batch_size_1,
        #                  neurons_dim=CONFIG.neurons_dim_1, n_epochs=CONFIG.n_epochs_1)
        #
        # train_sets2, test_sets2 = FilesUtil.split_data_set(CONFIG.G2_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets2, gesture_id=str(2), batch_size=CONFIG.batch_size_2,
        #                  neurons_dim=CONFIG.neurons_dim_2, n_epochs=CONFIG.n_epochs_2)

        train_sets5, test_sets5 = FilesUtil.split_data_set(CONFIG.G5_PATH, CONFIG.train_space)
        create_rnn_model(train_data_sets=train_sets5, gesture_id=str(5), batch_size=CONFIG.batch_size_5,
                         neurons_dim=CONFIG.neurons_dim_5, n_epochs=CONFIG.n_epochs_5)

        # train_sets6, test_sets6 = FilesUtil.split_data_set(CONFIG.G6_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets6, gesture_id=str(6), batch_size=CONFIG.batch_size_6,
        #                  neurons_dim=CONFIG.neurons_dim_6, n_epochs=CONFIG.n_epochs_6)
        #
        # train_sets7, test_sets7 = FilesUtil.split_data_set(CONFIG.G7_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets7, gesture_id=str(7), batch_size=CONFIG.batch_size_7,
        #                  neurons_dim=CONFIG.neurons_dim_7, n_epochs=CONFIG.n_epochs_7)
        #
        # train_sets8, test_sets8 = FilesUtil.split_data_set(CONFIG.G8_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets8, gesture_id=str(8), batch_size=CONFIG.batch_size_8,
        #                  neurons_dim=CONFIG.neurons_dim_8, n_epochs=CONFIG.n_epochs_8)

    @staticmethod
    def evaluate_model(gesture_id):

        outer_list = list()
        for data_set in enumerate(CONFIG.UNI_GE_DATA_SETS):
            inner_list = list()

            for i in range(CONFIG.evaluation_runs):
                train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(data_set[1]), CONFIG.train_space)
                x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(gesture_id),
                                                  CONFIG.get_neurons_dim(gesture_id))
                mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
                mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
                mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
                inner_list.append(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
                # main_index = data_set[1]
                # result_dictionary[main_index][i] =
            outer_list.append(inner_list)
        return DataFrame(outer_list, index=[1, 2, 5, 6, 7, 8])

    # @staticmethod
    # def simulate_online_use(path):
    #     df = DataFrame(np.genfromtxt(path, skip_header=True, delimiter=',',
    #                                  usecols=(CONFIG.col_id_1, CONFIG.col_id_2, CONFIG.col_id_3)))
    #
    #     # number of windows
    #     n = int(len(df) / 150)
    #     count = 0
    #     for i in range(n):
    #         print(df[count:count + 150])

    @staticmethod
    def test():
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(1), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
            mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
            mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        # print("----------------------------------------------------")
        # for i in range(CONFIG.evaluation_runs):
        #     train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(2), CONFIG.train_space)
        #     x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
        #                                       CONFIG.get_neurons_dim(1))
        #     mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
        #     mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
        #     mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
        #     print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        print("----------------------------------------------------")
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(5), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
            mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
            mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        # print("----------------------------------------------------")
        # for i in range(CONFIG.evaluation_runs):
        #     train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(6), CONFIG.train_space)
        #     x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
        #                                       CONFIG.get_neurons_dim(1))
        #     mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
        #     mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
        #     mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
        #     print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        # print("----------------------------------------------------")
        # for i in range(CONFIG.evaluation_runs):
        #
        #     train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(7), CONFIG.train_space)
        #     x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
        #                                       CONFIG.get_neurons_dim(1))
        #     mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
        #     mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
        #     mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
        #     print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        # print("----------------------------------------------------")
        # for i in range(CONFIG.evaluation_runs):
        #     train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(8), CONFIG.train_space)
        #     x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
        #                                       CONFIG.get_neurons_dim(1))
        #     mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
        #     mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
        #     mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
        #     print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))


# Launcher.train_models()
# Launcher.launch_evaluation()
Launcher.test()
# test1 = FilesUtil.generate_data_set(os.path.join(CONFIG.G1_PATH))
# test2 = FilesUtil.generate_data_set(os.path.join(CONFIG.G5_PATH))
#
# for list1, list2 in zip(test1, test2):
#     for v1, v2 in zip(list1.values, list2.values):
#         print(v1[0] - v2[0])
#         print(v1[1] - v2[1])
#         print(v1[2] - v2[2])
#         print("----------------------------------------------")
#     print("--------------------GESTURE-----------------------")
# # for j in enumerate(val):
# # print(j)
