import os
import csv
import glob
import numpy as np
import pandas as pd
from enum import Enum
import random
from matplotlib import pyplot as plt

from config import CONFIG


# todo : general turn unnecessary static method and turn them to instance method

class FilesUtil(object):

    @staticmethod
    def get_percent_diff(a, b):
        return (abs(a - b) / max(abs(a), abs(b))) * 100
        # return (a / b) * 100

    @staticmethod
    def reject_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    @staticmethod
    def split_data_set(path, train_dim):
        data_set = FilesUtil.generate_data_set(path)
        train_data_sets = data_set[:train_dim]
        temp_data_sets = data_set[train_dim:]
        test_data_sets = temp_data_sets[:train_dim]
        validation_data_sets = temp_data_sets[train_dim:]
        return train_data_sets, test_data_sets, validation_data_sets

    @staticmethod
    def print_current_path(path):
        print("CURRENT PATH: " + path)

    @staticmethod
    def get_current_path(path):
        return path

    @staticmethod
    def clean_csv_files(path):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file_path.endswith('csv'):
                    os.remove(file_path)

    @staticmethod
    def convert_folder_content_to_csv(path):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                text_file_path = os.path.join(subdir, file)
                id = 0
                in_txt = csv.reader(open(text_file_path, "r"), delimiter=' ')
                csv_file = text_file_path.replace('.txt', '.csv')
                out_csv = csv.writer(open(csv_file, "w"))
                count = 0
                lines = list()
                for line in in_txt:
                    line.append(count)
                    count += 1
                    lines.append(line)
                # print(line)
                for line in lines:
                    out_csv.writerow(line)

    @staticmethod
    def generate_data_set(path):
        data = list()
        for subdir, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file_path.endswith('csv'):
                    # print(os.path.relpath(file_path))
                    df = pd.DataFrame(np.genfromtxt(file_path, skip_header=True, delimiter=',',
                                                    usecols=(CONFIG.col_id_1, CONFIG.col_id_2, CONFIG.col_id_3)))
                    data.append(df)
        return data

    @staticmethod
    def get_train_example(data_set, seq_len):
        # todo reshape with train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        # todo generalizzare nel caso il numero degli input è maggiore di 3
        df = pd.DataFrame(data_set)
        df = pd.concat([df, df.shift(-1)], axis=1)
        df = df.fillna(0)
        values = df.values
        x = values[:, 0:3]
        y = values[:, 3:6]
        return x, y, seq_len

    @staticmethod
    def cut_and_reshape(data_set, cut):
        df = pd.DataFrame(data_set[:cut])
        # df = df.shift(-1)
        # df = df.fillna(0)
        x_test = df.values
        return np.array(x_test.reshape(-1, cut, x_test.shape[1]))

    @staticmethod
    def reshape(data_set):
        df = pd.DataFrame(data_set)
        x_test = df.values
        return np.array(x_test.reshape(-1, len(x_test), x_test.shape[1]))

    @staticmethod
    def feed_next_batch(data_sets, batch_size, max_size):
        c = 0
        batch_x = list()
        batch_y = list()
        sequences = list()
        # shuffled_data_set = random.shuffle(data_sets)
        shuffled_data_set = random.sample(data_sets, len(data_sets))
        # todo eventually pick a set of example with "random.sample(list, n_batch)
        # todo instead of picking the first n-batch from the full list
        while c < batch_size:
            data, size = FilesUtil.pad_example(FilesUtil.get_next_file(c, shuffled_data_set), max_size)
            x, y, seq = FilesUtil.get_train_example(data, size)
            # x = x.reshape((-1, x.shape[0], x.shape[1]))
            # y = y.reshape((-1, y.shape[0], y.shape[1]))
            batch_x.append(x)
            batch_y.append(y)
            sequences.append(seq)
            c += 1
        return batch_x, batch_y, sequences

    @staticmethod
    def pad_example(data_frame, max_steps):
        real_len = data_frame.shape[0]
        if real_len < max_steps:
            zeroes = pd.DataFrame(np.zeros((max_steps - real_len, data_frame.shape[1]), dtype=data_frame.dtypes))
            # print("added zeroes")
            df = pd.concat([data_frame, zeroes], axis=0)
            return df, real_len
        else:
            # print("cutted array")
            df = data_frame[:max_steps]
            return df, df.shape[0]

    @staticmethod
    def get_random_file(data_set):
        df = pd.DataFrame(random.SystemRandom().choice(data_set))
        return df

    @staticmethod
    def get_next_file(index, data_sets):
        return data_sets[index]

    @staticmethod
    def plot_series(series1, series2):
        # series1 = series1_r[:-1]
        # series2 = series2_r[1:]
        plt.figure(1)
        ax = plt.subplot(311)
        ax.set_title("x-data")
        ax.plot(np.arange(10), series1[:, 0], label="value")
        ax.plot(np.arange(10), series2[:, 0], label="prediction")
        ax.legend(loc='upper left')
        ay = plt.subplot(312)
        ay.plot(np.arange(10), series1[:, 1], label="value")
        ay.plot(np.arange(10), series2[:, 1], label="prediction")
        ay.legend(loc='upper left')
        ay.set_title("y-data")
        az = plt.subplot(313)
        az.plot(np.arange(10), series1[:, 2], label="value")
        az.plot(np.arange(10), series2[:, 2], label="prediction")
        az.legend(loc='upper left')
        az.set_title("z-data")
        plt.tight_layout()
        plt.show()
