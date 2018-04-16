from rnn import create_rnn_model, evaluate_rnn_model, get_mse
from file_utils import FilesUtil
from config import CONFIG
from pandas import DataFrame
import numpy as np

# FilesUtil.convert_folder_content_to_csv(FilesUtil.U1_PATH)
# FilesUtil.convert_folder_content_to_csv(FilesUtil.U2_PATH)
# FilesUtil.convert_folder_content_to_csv(FilesUtil.U3_PATH)
# FilesUtil.convert_folder_content_to_csv(FilesUtil.U4_PATH)
# FilesUtil.convert_folder_content_to_csv(FilesUtil.U5_PATH)
# FilesUtil.convert_folder_content_to_csv(FilesUtil.U6_PATH)

mse1List = list()
mse2List = list()
for i in range(10):
    train_sets1, test_sets1 = FilesUtil.split_data_set(FilesUtil.U1_PATH, CONFIG.train_space)
    train_sets2, test_sets2 = FilesUtil.split_data_set(FilesUtil.U6_PATH, CONFIG.train_space)

    x_new1, y_pre1 = evaluate_rnn_model(FilesUtil.get_random_file(test_sets1))
    x_new2, y_pre2 = evaluate_rnn_model(FilesUtil.get_random_file(test_sets2))

    mse2_x = get_mse(x_new2[0][:, 0], y_pre2[0][:, 0])
    mse1_x = get_mse(x_new1[0][:, 0], y_pre1[0][:, 0])
    # print('gesture 2 x:', mse2_x, ' - gesture 1 x:', mse1_x)
    mse2_y = get_mse(x_new2[0][:, 1], y_pre2[0][:, 1])
    mse1_y = get_mse(x_new1[0][:, 1], y_pre1[0][:, 1])
    # print('gesture 2 y:', mse2_y, ' - gesture 1 y:', mse1_y)
    mse2_z = get_mse(x_new2[0][:, 2], y_pre2[0][:, 2])
    mse1_z = get_mse(x_new1[0][:, 2], y_pre1[0][:, 2])
    # print('gesture 2 z:', mse2_z, ' - gesture 1 z:', mse1_z)
    mse1 = np.array([mse1_x, mse1_y, mse1_z])
    mse2 = np.array([mse2_x, mse2_y, mse2_z])
    mse1List.append(mse1)
    mse2List.append(mse2)

# for i, val in enumerate(mse1List):
print(np.mean(mse1List, axis=0))
print(np.mean(mse2List, axis=0))
