from matplotlib import pyplot as plt
from threading import Thread

from rnn import create_rnn_model, evaluate_rnn_model, get_mse
from file_utils import FilesUtil
from config import CONFIG
from pandas import DataFrame
from collections import defaultdict
import numpy as np
import os
from itertools import islice
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

        train_sets1, test_sets1, validation_sets1 = FilesUtil.split_data_set(CONFIG.G1_PATH, CONFIG.train_space)
        create_rnn_model(train_data_sets=train_sets1, validation_data_sets=validation_sets1,
                         gesture_id=str(1),
                         batch_size=CONFIG.batch_size_1,
                         neurons_dim=CONFIG.neurons_dim_1, n_epochs=CONFIG.n_epochs_1)

        # train_sets2, test_sets2, validation_sets2 = FilesUtil.split_data_set(CONFIG.G2_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets2, validation_data_sets=validation_sets2,
        #                  gesture_id=str(2),
        #                  batch_size=CONFIG.batch_size_2,
        #                  neurons_dim=CONFIG.neurons_dim_2, n_epochs=CONFIG.n_epochs_2)
        #
        # train_sets5, test_sets5, validation_sets5 = FilesUtil.split_data_set(CONFIG.G5_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets5, validation_data_sets=validation_sets5,
        #                  gesture_id=str(5),
        #                  batch_size=CONFIG.batch_size_5,
        #                  neurons_dim=CONFIG.neurons_dim_5, n_epochs=CONFIG.n_epochs_5)
        #
        # train_sets6, test_sets6, validation_sets6 = FilesUtil.split_data_set(CONFIG.G6_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets6, validation_data_sets=validation_sets6,
        #                  gesture_id=str(6), batch_size=CONFIG.batch_size_6,
        #                  neurons_dim=CONFIG.neurons_dim_6, n_epochs=CONFIG.n_epochs_6)
        # #
        # train_sets7, test_sets7, validation_sets7 = FilesUtil.split_data_set(CONFIG.G7_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets7, validation_data_sets=validation_sets7,
        #                  gesture_id=str(7), batch_size=CONFIG.batch_size_7,
        #                  neurons_dim=CONFIG.neurons_dim_7, n_epochs=CONFIG.n_epochs_7)
        #
        # train_sets8, test_sets8, validation_sets8 = FilesUtil.split_data_set(CONFIG.G8_PATH, CONFIG.train_space)
        # create_rnn_model(train_data_sets=train_sets8, validation_data_sets=validation_sets8,
        #                  gesture_id=str(8),
        #                  batch_size=CONFIG.batch_size_8,
        #                  neurons_dim=CONFIG.neurons_dim_8, n_epochs=CONFIG.n_epochs_8)

    @staticmethod
    def evaluate_model(gesture_id):

        outer_list = list()
        for data_set in enumerate(CONFIG.UNI_GE_DATA_SETS):
            inner_list = list()

            for i in range(CONFIG.evaluation_runs):
                train_set, test_set, validation = FilesUtil.split_data_set(CONFIG.get_path(data_set[1]),
                                                                           CONFIG.train_space)
                x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(gesture_id),
                                                  CONFIG.get_neurons_dim(gesture_id))
                mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
                mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
                mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
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
            train_set, test_set, valid = FilesUtil.split_data_set(CONFIG.get_path(1), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
            mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
            mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        print("----------------------------------------------------")
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set, valid = FilesUtil.split_data_set(CONFIG.get_path(2), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
            mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
            mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        print("----------------------------------------------------")
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set, valid = FilesUtil.split_data_set(CONFIG.get_path(5), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
            mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
            mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        print("----------------------------------------------------")
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set, valid = FilesUtil.split_data_set(CONFIG.get_path(6), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
            mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
            mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        print("----------------------------------------------------")
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set, valid = FilesUtil.split_data_set(CONFIG.get_path(7), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
            mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
            mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
        print("----------------------------------------------------")
        for i in range(CONFIG.evaluation_runs):
            train_set, test_set, valid = FilesUtil.split_data_set(CONFIG.get_path(8), CONFIG.train_space)
            x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(1),
                                              CONFIG.get_neurons_dim(1))
            mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
            mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
            mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
            print(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))

    @staticmethod
    def online_test(path, gesture_id):
        data_set = FilesUtil.generate_data_set_from_file(path)
        start = 0
        end = 149
        errors = list()
        for i in range(len(data_set)):

            if end < len(data_set):
                current = data_set[start:end]
                x_new, y_pre = evaluate_rnn_model(current.values, str(gesture_id), CONFIG.get_neurons_dim(gesture_id))
                mse_x = get_mse(x_new[0][1:, 0], y_pre[0][:-1, 0])
                mse_y = get_mse(x_new[0][1:, 1], y_pre[0][:-1, 1])
                mse_z = get_mse(x_new[0][1:, 2], y_pre[0][:-1, 2])
                errors.append(np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z))
                current, start, end = FilesUtil.get_next_window(current, dimension=150, start=start, end=end)
                print("done " + str(i) + "  of " + len(data_set).__str__() + " error " + str(
                    np.sqrt(mse_x * mse_x + mse_y * mse_y + mse_z * mse_z)) + " gesture " + str(gesture_id))
        return errors

    @staticmethod
    def update_sliding(data,window):

        #create a windos of nan - passed as argument





        return ""

    @staticmethod
    def detect(series):
        threshold = 0.2
        gesture_duration = 150
        picks = {}




        # print(series[212:444])
        # print(v[154 * 4:179 * 4])
        # print(v[232 * 4:254 * 4])
        # print(v[284 * 4:326 * 4])
        # print(v[342 * 4:359 * 4])
        # print(v[366 * 4:377 * 4])
        # print(v[378 * 4:386 * 4])
        # print(v[396 * 4:421 * 4])
        # print(v[439 * 4:476 * 4])
        # print(v[501 * 4:542 * 4])


e1 = FilesUtil.load_result_file('err1.p')
e2 = FilesUtil.load_result_file('err2.p')
e5 = FilesUtil.load_result_file('err5.p')
e6 = FilesUtil.load_result_file('err6.p')
e7 = FilesUtil.load_result_file('err7.p')
e8 = FilesUtil.load_result_file('err8.p')

# Launcher.detect(series=e1)



# FilesUtil.convert_folder_content_to_csv(CONFIG.ONLINE_DATA_SET)
# Launcher.train_models()'1122334455.txt.txt'
# err1 = Launcher.online_test(os.path.join(os.getcwd(), 'online_data/1122334455.txt'), 1)
# err2 = Launcher.online_test(os.path.join(os.getcwd(), 'online_data/1122334455.txt'), 2)
# err5 = Launcher.online_test(os.path.join(os.getcwd(), 'online_data/1122334455.txt'), 5)
# err6 = Launcher.online_test(os.path.join(os.getcwd(), 'online_data/1122334455.txt'), 6)
# err7 = Launcher.online_test(os.path.join(os.getcwd(), 'online_data/1122334455.txt'), 7)
# err8 = Launcher.online_test(os.path.join(os.getcwd(), 'online_data/1122334455.txt'), 8)

# _len = len(err1)

# plt.figure(1)
# a1 = plt.subplot(611)
# a1.set_title("x-data")
# a1.plot(np.arange(len(err1)), err1)
# a2 = plt.subplot(612)
# a2.plot(np.arange(len(err2)), err2)
# a2.set_title("y-data")
# a5 = plt.subplot(613)
# a5.plot(np.arange(len(err3)), err3)
# a5.set_title("z-data")
# a6 = plt.subplot(614)
# a6.set_title("x-data")
# a6.plot(np.arange(len(err4)), err4)
# a7 = plt.subplot(615)
# a7.plot(np.arange(len(err5)), err5)
# a7.set_title("y-data")
# a8 = plt.subplot(616)
# a8.plot(np.arange(len(err6)), err6)
# a8.set_title("z-data")
# plt.tight_layout()
# plt.show()

# res1 = dict()
# res2 = dict()
# res3 = dict()
# res4 = dict()
# res5 = dict()
# res6 = dict()

# FilesUtil.save_results_to_file(err1, 1)
# FilesUtil.save_results_to_file(err2, 2)
# FilesUtil.save_results_to_file(err5, 5)
# FilesUtil.save_results_to_file(err6, 6)
# FilesUtil.save_results_to_file(err7, 7)
# FilesUtil.save_results_to_file(err8, 8)

# check for the proper mse in the proper interval
# for i, v in enumerate(err1):
#     res1['1'] = v[53 * 4:111 * 4]
#     res1['2'] = v[154 * 4:179 * 4]
#     res1['3'] = v[232 * 4:254 * 4]
#     res1['4'] = v[284 * 4:326 * 4]
#     res1['5'] = v[342 * 4:359 * 4]
#     res1['6'] = v[366 * 4:377 * 4]
#     res1['7'] = v[378 * 4:386 * 4]
#     res1['8'] = v[396 * 4:421 * 4]
#     res1['9'] = v[439 * 4:476 * 4]
#     res1['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err2):
#     res2['1'] = v[53 * 4:111 * 4]
#     res2['2'] = v[154 * 4:179 * 4]
#     res2['3'] = v[232 * 4:254 * 4]
#     res2['4'] = v[284 * 4:326 * 4]
#     res2['5'] = v[342 * 4:359 * 4]
#     res2['6'] = v[366 * 4:377 * 4]
#     res2['7'] = v[378 * 4:386 * 4]
#     res2['8'] = v[396 * 4:421 * 4]
#     res2['9'] = v[439 * 4:476 * 4]
#     res2['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err3):
#     res3['1'] = v[53 * 4:111 * 4]
#     res3['2'] = v[154 * 4:179 * 4]
#     res3['3'] = v[232 * 4:254 * 4]
#     res3['4'] = v[284 * 4:326 * 4]
#     res3['5'] = v[342 * 4:359 * 4]
#     res3['6'] = v[366 * 4:377 * 4]
#     res3['7'] = v[378 * 4:386 * 4]
#     res3['8'] = v[396 * 4:421 * 4]
#     res3['9'] = v[439 * 4:476 * 4]
#     res3['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err4):
#     res4['1'] = v[53 * 4:111 * 4]
#     res4['2'] = v[154 * 4:179 * 4]
#     res4['3'] = v[232 * 4:254 * 4]
#     res4['4'] = v[284 * 4:326 * 4]
#     res4['5'] = v[342 * 4:359 * 4]
#     res4['6'] = v[366 * 4:377 * 4]
#     res4['7'] = v[378 * 4:386 * 4]
#     res4['8'] = v[396 * 4:421 * 4]
#     res4['9'] = v[439 * 4:476 * 4]
#     res4['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err5):
#     res5['1'] = v[53 * 4:111 * 4]
#     res5['2'] = v[154 * 4:179 * 4]
#     res5['3'] = v[232 * 4:254 * 4]
#     res5['4'] = v[284 * 4:326 * 4]
#     res5['5'] = v[342 * 4:359 * 4]
#     res5['6'] = v[366 * 4:377 * 4]
#     res5['7'] = v[378 * 4:386 * 4]
#     res5['8'] = v[396 * 4:421 * 4]
#     res5['9'] = v[439 * 4:476 * 4]
#     res5['10'] = v[501 * 4:542 * 4]

# for i, v in enumerate(err6):
#     res6['1'] = v[53 * 4:111 * 4]
#     res6['2'] = v[154 * 4:179 * 4]
#     res6['3'] = v[232 * 4:254 * 4]
#     res6['4'] = v[284 * 4:326 * 4]
#     res6['5'] = v[342 * 4:359 * 4]
#     res6['6'] = v[366 * 4:377 * 4]
#     res6['7'] = v[378 * 4:386 * 4]
#     res6['8'] = v[396 * 4:421 * 4]
#     res6['9'] = v[439 * 4:476 * 4]
#     res6['10'] = v[501 * 4:542 * 4]

# print(res2)

# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      1)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      2)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      5)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      6)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      7)
# Launcher.online_test('C:\\Users\\Davide\\Documents\Projects\\sofar_gesture_recognition_tf\\online_data\\1x2x5x6x1.txt',
#                      8)
# 'C:\\Users\\Davide\\Documents\\Projects\\sofar_gesture_recognition_tf\\online_data/1115556662.csv'
# Launcher.launch_evaluation()
# Launcher.test()
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
