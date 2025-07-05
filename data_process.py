# coding:utf-8

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time, datetime
import utils


def main():
    # extract station id list in Beijing
    df_airq = pd.read_csv('./data/microsoft_urban_air_data/airquality.csv')
    station_id_list = np.unique(df_airq['station_id'])[:36]     # first 36 stations are in Beijing
    print(station_id_list)

    # Calculate the influence degree (defined as the Pearson correlation coefficient) between the center station and other stations
    r_thred = 0.85 #0.85
    center_station_id = 1013
    station_id_related_list = []
    df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(center_station_id))
    v_list_1 = list(df_one_station['PM25_Concentration'])
    for station_id_other in station_id_list:
        df_one_station_other = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id_other))
        v_list_2 = list(df_one_station_other['PM25_Concentration'])
        r, p = stats.pearsonr(v_list_1, v_list_2)
        if r > r_thred:
            station_id_related_list.append(station_id_other)
        print('{}  {}  {:.3f}'.format(center_station_id, station_id_other, r))
    print(len(station_id_related_list), station_id_related_list)

    # generate x and y
    # x_shape: [example_count, num_releated, seq_step, feat_size]
    # y_shape: [example_count,]
    print('Center station: {}\nRelated stations: {}'.format(center_station_id, station_id_related_list))
    feat_names = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
                  'weather', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    x_length = 3
    y_length = 1 # 预测步长
    y_step = 1 # =1,连续地逐步预测未来 24 小时的数据
    x = []
    y = []
    for station_id in station_id_related_list:
        df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id))
        x_one = []
        for start_id in range(0, len(df_one_station)-x_length-y_length+1-y_step+1, y_length):
            x_data = np.array(df_one_station[feat_names].iloc[start_id: start_id+x_length])
            y_list = np.array(df_one_station['PM25_Concentration'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
            if np.isnan(x_data).any() or np.isnan(y_list).any():
                continue
            x_one.append(x_data)
            if station_id == center_station_id:
                y.append(np.mean(y_list))
        if len(x_one) <= 0:
            continue

        # x_one_stacked = np.stack(x_one, axis=0)
        # x.append(x_one_stacked)
        x_one = np.array(x_one)
        x.append(x_one)
        # Concatenate along feat_size dimension
        # x = np.concatenate(x_one, axis=2)
        # x.append(x_one)
        print('station_id: {}  x_shape: {}'.format(station_id, x_one.shape))

    # x = np.concatenate(x, axis=2)
    x = np.array(x)
    x = x.transpose((1, 0, 2, 3))
    y = np.array(y)
    print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))

    # Save the four dimensional data as pickle file
    utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    print('x_shape: {}\ny_shape: {}'.format(x.shape, y.shape))

    #
    #     x_one_stacked = np.stack(x_one, axis=0)
    #     x.append(x_one_stacked)
    #     print('station_id: {}  x_shape: {}'.format(station_id, x_one_stacked.shape))
    #
    # x = np.concatenate(x, axis=2)
    # x = np.array(x)
    # y = np.array(y)
    # print('x_shape: {}  y_shape: {}'.format(x.shape, y.shape))
    #
    # # Save the four dimensional data as pickle file
    # utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
    # utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
    # print('x_shape: {}\ny_shape: {}'.format(x.shape, y.shape))

if __name__ == '__main__':
    main()


# #
# # #
# # ###########只有中心站点
# # import numpy as np
# # import pandas as pd
# # import utils
# #
# #
# # def main():
# #     # # 读取数据
# #     # df_airq = pd.read_csv('./data/microsoft_urban_air_data/airquality.csv')
# #     # station_id_list = np.unique(df_airq['station_id'])[:36]  # 选取前 36 个站
# #
# #     # 设置中心站点
# #     center_station_id = 1013
# #     df_center_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(center_station_id))
# #
# #     # 特征名
# #     feat_names = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration',
# #                   'CO_Concentration', 'O3_Concentration', 'SO2_Concentration',
# #                   'weather', 'temperature', 'pressure', 'humidity',
# #                   'wind_speed', 'wind_direction']
# #
# #     x_length = 3  # 每个样本的时间步长
# #     y_length = 1  # 预测步长
# #     y_step = 1  # 步长
# #     x = []
# #     y = []
# #
# #     # 直接处理中心站点数据
# #     for start_id in range(0, len(df_center_station) - x_length - y_length + 1 - y_step + 1, y_step):
# #         x_data = np.array(df_center_station[feat_names].iloc[start_id:start_id + x_length])
# #         y_list = np.array(df_center_station['PM25_Concentration'].iloc[
# #                           start_id + x_length + y_step - 1:start_id + x_length + y_length + y_step - 1])
# #
# #         if np.isnan(x_data).any() or np.isnan(y_list).any():
# #             continue
# #
# #         x.append(x_data)  # 添加到列表中
# #         y.append(np.mean(y_list))
# #
# #     # 转换为 NumPy 数组
# #     x = np.array(x)  # 这里的 x 将是二维的，形状为 (batch_size, 24, 12)
# #     y = np.array(y)
# #
# #     # 输出 x 的维度
# #     print('x 的维度:', x.shape)
# #
# #     # 保存数据
# #     utils.save_pickle('./data/xy/x_{}.pkl'.format(center_station_id), x)
# #     utils.save_pickle('./data/xy/y_{}.pkl'.format(center_station_id), y)
# #
# #
# # if __name__ == '__main__':
# #     main()
# #
#
# ##############
# # coding:utf-8
#
# import numpy as np
# import pandas as pd
# from scipy import stats
# import matplotlib.pyplot as plt
# # import seaborn as sns
# from sklearn import preprocessing
# import time, datetime
# import utils
#
#
# def get_season(date_str):
#     """根据日期判断季节"""
#     date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
#     month = date.month
#     if month in [3, 4, 5]:
#         return 'spring'
#     elif month in [6, 7, 8]:
#         return 'summer'
#     elif month in [9, 10, 11]:
#         return 'autumn'
#     else:
#         return 'winter'
#
#
# def process_season_data(df_one_station, station_id, center_station_id, feat_names, x_length, y_length, y_step, season):
#     """处理指定季节的数据"""
#     # 添加season列
#     df_one_station['season'] = df_one_station['time'].apply(get_season)
#     # 筛选特定季节的数据
#     df_season = df_one_station[df_one_station['season'] == season].copy()
#
#     x_one = []
#     y_season = []
#
#     for start_id in range(0, len(df_season) - x_length - y_length + 1 - y_step + 1, y_length):
#         x_data = np.array(df_season[feat_names].iloc[start_id:start_id + x_length])
#         y_list = np.array(df_season['PM25_Concentration'].iloc[
#                           start_id + x_length + y_step - 1:start_id + x_length + y_length + y_step - 1])
#
#         if np.isnan(x_data).any() or np.isnan(y_list).any():
#             continue
#
#         x_one.append(x_data)
#         if station_id == center_station_id:
#             y_season.append(np.mean(y_list))
#
#     return np.array(x_one) if len(x_one) > 0 else None, np.array(y_season) if len(y_season) > 0 else None
#
#
# def main():
#     # Extract station id list in Beijing
#     df_airq = pd.read_csv('./data/microsoft_urban_air_data/airquality.csv')
#     station_id_list = np.unique(df_airq['station_id'])[:36]  # first 36 stations are in Beijing
#     print(station_id_list)
#
#     # Calculate the influence degree
#     r_thred = 0.85
#     center_station_id = 1013
#     station_id_related_list = []
#     df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(center_station_id))
#     v_list_1 = list(df_one_station['PM25_Concentration'])
#
#     for station_id_other in station_id_list:
#         df_one_station_other = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id_other))
#         v_list_2 = list(df_one_station_other['PM25_Concentration'])
#         r, p = stats.pearsonr(v_list_1, v_list_2)
#         if r > r_thred:
#             station_id_related_list.append(station_id_other)
#         print('{}  {}  {:.3f}'.format(center_station_id, station_id_other, r))
#
#     print(len(station_id_related_list), station_id_related_list)
#     print('Center station: {}\nRelated stations: {}'.format(center_station_id, station_id_related_list))
#
#     feat_names = ['PM25_Concentration', 'PM10_Concentration', 'NO2_Concentration', 'CO_Concentration',
#                   'O3_Concentration', 'SO2_Concentration', 'weather', 'temperature',
#                   'pressure', 'humidity', 'wind_speed', 'wind_direction']
#
#     x_length = 3
#     y_length = 1
#     y_step = 1
#
#     # 对四个季节分别处理数据
#     seasons = ['spring', 'summer', 'autumn', 'winter']
#     for season in seasons:
#         x_season = []
#         y_season = []
#
#         for station_id in station_id_related_list:
#             df_one_station = pd.read_csv('./data/stations_data/df_station_{}.csv'.format(station_id))
#             x_one_season, y_one_season = process_season_data(
#                 df_one_station, station_id, center_station_id, feat_names,
#                 x_length, y_length, y_step, season
#             )
#
#             if x_one_season is not None:
#                 x_season.append(x_one_season)
#                 if station_id == center_station_id:
#                     y_season = y_one_season
#                 print(f'station_id: {station_id}  {season}_x_shape: {x_one_season.shape}')
#
#         if len(x_season) > 0:
#             x_season = np.array(x_season)
#             x_season = x_season.transpose((1, 0, 2, 3))
#             y_season = np.array(y_season)
#             print(f'{season}_x_shape: {x_season.shape}  {season}_y_shape: {y_season.shape}')
#
#             # 保存每个季节的数据
#             utils.save_pickle(f'./data/xy/x_{center_station_id}_{season}.pkl', x_season)
#             utils.save_pickle(f'./data/xy/y_{center_station_id}_{season}.pkl', y_season)
#
#
# if __name__ == '__main__':
#     main()
#
