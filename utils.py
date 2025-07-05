# coding:utf-8

import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_ids_for_tvt():
    train_ids = []
    valid_ids = []
    test_ids = []
    days_in_months = [31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30-1]  # May to April
    start_id = 0
    for i in range(len(days_in_months)):
        days = days_in_months[i]
        split_id_0 = start_id
        split_id_1 = start_id + int(days * 24 * 0.6)
        split_id_2 = start_id + int(days * 24 * 0.8)
        split_id_3 = start_id + int(days * 24)
        train_ids.extend(np.arange(split_id_0, split_id_1, 1))# split_id_1gai2
        valid_ids.extend(np.arange(split_id_1, split_id_2, 1))
        test_ids.extend(np.arange(split_id_2, split_id_3, 1))
        start_id += int(days * 24)
    # split_id_0 = start_id
    # split_id_1 = int(121 * 0.6)
    # split_id_2 = int(121 * 0.8)
    # split_id_3 = 121
    # train_ids.extend(np.arange(split_id_0, split_id_1, 1))
    # valid_ids.extend(np.arange(split_id_1, split_id_2, 1))
    # test_ids.extend(np.arange(split_id_2, split_id_3, 1))

    return train_ids, valid_ids, test_ids


def load_data(f_x, f_y):
    x = load_pickle(f_x)
    y = load_pickle(f_y)
    y = np.array(y[:, np.newaxis])
    if len(x.shape) == 3:
        ss = preprocessing.StandardScaler()
        for i in range(x.shape[-1]):
            ss.fit(x[:, :, i])
            x[:, :, i] = ss.transform(x[:, :, i])
    train_ids, valid_ids, test_ids = get_ids_for_tvt()
    x_train = x[train_ids]
    y_train = y[train_ids]
    x_valid = x[valid_ids]
    y_valid = y[valid_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]

    print('x_shape: {}  y_shape: {}\nx_train_shape: {}  y_train_shape: {}  x_valid_shape: {}  y_valid_shape: {}  x_test_shape: {}  y_test_shape: {}\n'
          .format(x.shape, y.shape, x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape))
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_param_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num



#
#
# #############
#
# # coding:utf-8
#
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn import preprocessing
#
#
# def save_pickle(filename, data):
#     """保存数据到pickle文件"""
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)
#
#
# def load_pickle(name):
#     """从pickle文件加载数据"""
#     with open(name, 'rb') as f:
#         data = pickle.load(f)
#     return data
#
#
# def get_ids_for_tvt(data_length):
#     """根据数据长度生成训练、验证、测试集的索引"""
#     train_ids = []
#     valid_ids = []
#     test_ids = []
#     split_id_1 = int(data_length * 0.6)  # 60% 用于训练集
#     split_id_2 = int(data_length * 0.8)  # 80% 用于验证集
#     train_ids = np.arange(0, split_id_1)
#     valid_ids = np.arange(split_id_1, split_id_2)
#     test_ids = np.arange(split_id_2, data_length)
#     return train_ids, valid_ids, test_ids
#
#
# def load_data(f_x_list, f_y_list):
#     """
#     加载四季数据并生成训练、验证、测试集
#     f_x_list: 四个季节的 x 数据文件路径列表
#     f_y_list: 四个季节的 y 数据文件路径列表
#     返回：每个季节的训练、验证、测试集数据
#     """
#     x_all, y_all = [], []
#
#     # 分别加载四个季节的数据
#     for f_x, f_y in zip(f_x_list, f_y_list):
#         x = load_pickle(f_x)
#         y = load_pickle(f_y)
#         y = np.array(y[:, np.newaxis])  # 确保 y 形状兼容
#
#         # 标准化每个季节的特征
#         if len(x.shape) == 3:
#             ss = preprocessing.StandardScaler()
#             for i in range(x.shape[-1]):
#                 ss.fit(x[:, :, i])
#                 x[:, :, i] = ss.transform(x[:, :, i])
#
#         # 根据当前季节的数据长度生成索引
#         train_ids, valid_ids, test_ids = get_ids_for_tvt(len(x))
#
#         # 划分训练、验证和测试集
#         x_train, x_valid, x_test = x[train_ids], x[valid_ids], x[test_ids]
#         y_train, y_valid, y_test = y[train_ids], y[valid_ids], y[test_ids]
#
#         # 存储到列表中
#         x_all.append((x_train, x_valid, x_test))
#         y_all.append((y_train, y_valid, y_test))
#
#     return x_all, y_all
#
#
# def get_param_number(net):
#     """计算模型的参数数量"""
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return total_num, trainable_num
#
#
# def main():
#     # 假设您有四个季节的 x 和 y pickle 文件路径列表
#     f_x_list = ['./data/xy/x_1013_spring.pkl', './data/xy/x_1013_summer.pkl',
#                 './data/xy/x_1013_autumn.pkl', './data/xy/x_1013_winter.pkl']
#     f_y_list = ['./data/xy/y_1013_spring.pkl', './data/xy/y_1013_summer.pkl',
#                 './data/xy/y_1013_autumn.pkl', './data/xy/y_1013_winter.pkl']
#
#     # 加载数据
#     x_all, y_all = load_data(f_x_list, f_y_list)
#
#     # 显示每个季节的训练、验证、测试集数据的形状
#     seasons = ['spring', 'summer', 'autumn', 'winter']
#     for i, season in enumerate(seasons):
#         x_train, x_valid, x_test = x_all[i]
#         y_train, y_valid, y_test = y_all[i]
#         print(f"{season} - x_train: {x_train.shape}, y_train: {y_train.shape}")
#         print(f"{season} - x_valid: {x_valid.shape}, y_valid: {y_valid.shape}")
#         print(f"{season} - x_test: {x_test.shape}, y_test: {y_test.shape}")
#
#
# if __name__ == '__main__':
#     main()
