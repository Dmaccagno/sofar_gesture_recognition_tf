from rnn import create_rnn_model, evaluate_rnn_model, get_mse
from file_utils import FilesUtil
from config import CONFIG
from pandas import DataFrame
import numpy as np
import os


# train_sets1, test_sets1 = FilesUtil.split_data_set(CONFIG.G1_PATH, CONFIG.train_space)
# create_rnn_model(train_data_sets=train_sets1, gesture_id=str(1))
#
# train_sets2, test_sets2 = FilesUtil.split_data_set(CONFIG.G2_PATH, CONFIG.train_space)
# create_rnn_model(train_data_sets=train_sets2, gesture_id=str(2))
#
# train_sets5, test_sets5 = FilesUtil.split_data_set(CONFIG.G5_PATH, CONFIG.train_space)
# create_rnn_model(train_data_sets=train_sets5, gesture_id=str(5))
#
# train_sets6, test_sets6 = FilesUtil.split_data_set(CONFIG.G6_PATH, CONFIG.train_space)
# create_rnn_model(train_data_sets=train_sets6, gesture_id=str(6))
#
# train_sets7, test_sets7 = FilesUtil.split_data_set(CONFIG.G7_PATH, CONFIG.train_space)
# create_rnn_model(train_data_sets=train_sets7, gesture_id=str(7))
#
# train_sets8, test_sets8 = FilesUtil.split_data_set(CONFIG.G8_PATH, CONFIG.train_space)
# create_rnn_model(train_data_sets=train_sets8, gesture_id=str(8))


class Launcher(object):

    @staticmethod
    def evaluate_model(gesture_id):
        result_dictionary = {}
        x_list = list()
        y_list = list()
        z_list = list()
        for data_set in enumerate(CONFIG.UNI_GE_DATA_SETS):
            max_x = -1
            max_y = -1
            max_z = -1
            for i in range(10):
                train_set, test_set = FilesUtil.split_data_set(CONFIG.get_path(data_set[1]), CONFIG.train_space)
                x_new, y_pre = evaluate_rnn_model(FilesUtil.get_random_file(test_set), str(gesture_id))
                mse_x = get_mse(x_new[0][:, 0], y_pre[0][:, 0])
                mse_y = get_mse(x_new[0][:, 1], y_pre[0][:, 1])
                mse_z = get_mse(x_new[0][:, 2], y_pre[0][:, 2])
                # x_list.append(mse_x)
                # y_list.append(mse_y)
                # z_list.append(mse_z)
                if max_x < mse_x:
                    max_x = mse_x
                if max_y < mse_y:
                    max_y = mse_y
                if max_z < mse_z:
                    max_z = mse_z
            result_dictionary[data_set[1]] = np.sqrt(max_x*max_x + max_y*max_y + max_z*max_z)
        return result_dictionary


print("-------------------------- gesture 1 --------------------------")
result = Launcher.evaluate_model(1)
for i, v in result.items():
    print('gesture ' + str(i), v)
print("-------------------------- gesture 2 --------------------------")
result = Launcher.evaluate_model(2)
for i, v in result.items():
    print('gesture ' + str(i), v)
print("-------------------------- gesture 5 --------------------------")
result = Launcher.evaluate_model(5)
for i, v in result.items():
    print('gesture ' + str(i), v)
print("-------------------------- gesture 6 --------------------------")
result = Launcher.evaluate_model(6)
for i, v in result.items():
    print('gesture ' + str(i), v)
print("-------------------------- gesture 7 --------------------------")
result = Launcher.evaluate_model(7)
for i, v in result.items():
    print('gesture ' + str(i), v)
print("-------------------------- gesture 8 --------------------------")
result = Launcher.evaluate_model(8)
for i, v in result.items():
    print('gesture ' + str(i), v)

# mse1List = list()
# mse2List = list()
#
# max2X = -1
# max2Y = -1
# max2Z = -1
# max1X = -1
# max1Y = -1
# max1Z = -1
#
# for i in range(10):
#     train_sets1, test_sets1 = FilesUtil.split_data_set(CONFIG.G1_PATH, CONFIG.train_space)
#     train_sets2, test_sets2 = FilesUtil.split_data_set(CONFIG.G2_PATH, CONFIG.train_space)
#
#     x_new1, y_pre1 = evaluate_rnn_model(FilesUtil.get_random_file(test_sets1), str(1))
#     x_new2, y_pre2 = evaluate_rnn_model(FilesUtil.get_random_file(test_sets2), str(1))
#
#     mse2_x = get_mse(x_new2[0][:, 0], y_pre2[0][:, 0])
#     mse1_x = get_mse(x_new1[0][:, 0], y_pre1[0][:, 0])
#     # print('gesture 2 x:', mse2_x, ' - gesture 1 x:', mse1_x)
#     mse2_y = get_mse(x_new2[0][:, 1], y_pre2[0][:, 1])
#     mse1_y = get_mse(x_new1[0][:, 1], y_pre1[0][:, 1])
#     # print('gesture 2 y:', mse2_y, ' - gesture 1 y:', mse1_y)
#     mse2_z = get_mse(x_new2[0][:, 2], y_pre2[0][:, 2])
#     mse1_z = get_mse(x_new1[0][:, 2], y_pre1[0][:, 2])
#     # print('gesture 2 z:', mse2_z, ' - gesture 1 z:', mse1_z)
#     if max1X < mse1_x:
#         max1X = mse1_x
#     if max1Y < mse1_y:
#         max1Y = mse1_y
#     if max1Z < mse1_z:
#         max1Z = mse1_z
#     if max2X < mse2_x:
#         max2X = mse2_x
#     if max2Y < mse2_y:
#         max2Y = mse2_y
#     if max2Z < mse2_z:
#         max2Z = mse2_z
#
# print(np.sqrt(np.square(max1X) + np.square(max1Y) + np.square(max1Z)))
# print(np.sqrt(np.square(max2X) + np.square(max2Y) + np.square(max2Z)))
