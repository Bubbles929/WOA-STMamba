# coding:utf-8

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
import models
import utils
import config as cfg
import matplotlib.pyplot as plt

def eval(net, x_test, y_test, plot=False):
    x_valid = x_test
    y_valid = y_test
    print('\nStart evaluating...\n')
    net.eval()
    rmse_train_list = []
    rmse_valid_list = []
    mae_valid_list = []
    y_valid_pred_final = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.L1Loss()
    h_state = None
    y_valid_pred_final = []
    mae_valid = 0.0
    cnt = 0
    for start in range(len(x_valid) - cfg.batch_size + 1):
        x_input_valid = torch.tensor(x_valid[start:start + cfg.batch_size], dtype=torch.float32)
        y_true_valid = torch.tensor(y_valid[start:start + cfg.batch_size], dtype=torch.float32)
        if cfg.model_name == 'RNN' or cfg.model_name == 'GRU' or cfg.model_name == 'gMamba':
            y_valid_pred, _h_state = net(x_input_valid, h_state)
        else:
            y_valid_pred = net(x_input_valid)
        y_valid_pred_final.extend(y_valid_pred.data.numpy())
        loss_valid = criterion(y_valid_pred, y_true_valid).data
        mae_valid_batch = loss_valid.numpy()
        # rmse_valid_batch = np.sqrt(mse_valid_batch)
        mae_valid += mae_valid_batch
        cnt += 1
    y_valid_pred_final = np.array(y_valid_pred_final).reshape((-1, 1))
    mae_valid = mae_valid / cnt
    rmse_valid = np.sqrt(metrics.mean_squared_error(y_valid, y_valid_pred_final))
    r2_valid = metrics.r2_score(y_valid, y_valid_pred_final)
    ##########

    # 将数据写入到一个文本文件
    file_path = 'predictions1.txt'

    with open(file_path, 'w') as file:
        file.write('Actual\tPredicted\n')  # 写入表头

        # 逐行写入数据
        for actual, predicted in zip(y_valid, y_valid_pred_final):
            file.write(f'{actual}\t{predicted}\n')

    print(f'Data has been written to {file_path}')

    # 创建一个新的图
    plt.figure(figsize=(10, 6))
    # 绘制 y_valid 的实际值折线
    plt.plot(y_valid, label='Actual', color='#F7BB97') #7ABBDB
    # 绘制 y_valid_pred_final 的预测值折线
    plt.plot(y_valid_pred_final, label='Predicted', color='#77DD77')#DCA7EB
    # 添加标题和标签
    plt.title('WOA-STMmaba-Winter')
    plt.xlabel('Sample')
    plt.ylabel('PM2.5 Concentration')
    # 添加图例和网格
    plt.legend()
    plt.grid(True)
    # 显示图表
    plt.show()
    # 创建一个新的图 - 散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_valid, y_valid_pred_final, color='#A2D5F2', alpha=0.6, edgecolors='w', s=40)   #84BA42   #FDBCB4 3#F4A460

    # 添加对角线，表示完全匹配的情况
    max_value = max(max(y_valid), max(y_valid_pred_final))
    plt.plot([0, max_value], [0, max_value], color='black', linestyle='--', linewidth=2)

    # 添加标题和标签
    plt.title('WOA-STMmaba-Winter')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)

    # 添加网格和图例
    plt.gca().set_aspect('equal', adjustable='box')  # 使 X 轴和 Y 轴比例相等
    plt.show()

    ##########


    print('\nRMSE_valid: {:.4f}  MAE_valid: {:.4f}  R2_valid: {:.4f}\n'.format(rmse_valid, mae_valid, r2_valid))
    return rmse_valid, mae_valid, r2_valid


def main():
    # Hyper Parameters
    cfg.print_params()
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    # Load data
    print('\nLoading data...\n')
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y)

    # # 加载数据
    # x_all, y_all = utils.load_data(f_x_list=cfg.f_x, f_y_list=cfg.f_y)
    #
    # # 显示每个季节的训练、验证、测试集数据的形状
    # seasons = ['winter']  # 'spring', , 'autumn', 'winter'
    # for i, season in enumerate(seasons):
    #     x_train, x_valid, x_test = x_all[i]
    #     y_train, y_valid, y_test = y_all[i]

    # Generate model
    net = None
    if cfg.model_name == 'RNN':
        net = models.SimpleRNN(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'GRU':
        net = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'LSTM':
        net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
    elif cfg.model_name == 'TCN':
        net = models.TCN(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
    elif cfg.model_name == 'Mamba':
        net = models.SimpleMamba(d_model=cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                                 d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
    elif cfg.model_name == 'oMamba':
        net = models.oMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, out_in_channels=cfg.out_in_channels,
                            d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                            d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
    elif cfg.model_name == 'gMamba':
        # net = models.gMamba(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size,
        #                 num_layers=cfg.num_layers, in_channels=cfg.in_channels)
        # net = models.gMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, out_in_channels=cfg.out_in_channels,
        #                     output_size=cfg.output_size, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
        # net = models.gMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, out_in_channels=cfg.out_in_channels,
        #                     output_size=cfg.output_size, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
        #                     d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
        net = models.gMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, out_in_channels=cfg.out_in_channels,
                            d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                            d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))

    # Load model parameters
    net.load_state_dict(torch.load(cfg.model_save_pth))
    print(utils.get_param_number(net=net))

    # Evaluation
    eval(net, x_test, y_test)


if __name__ == '__main__':
    main()

