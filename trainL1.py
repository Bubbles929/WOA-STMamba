# coding:utf-8

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
import models
import utils
import config as cfg


def train(net, x_train, y_train, x_valid, y_valid, x_test, y_test, plot=False):
    mae_train_list = []
    mae_valid_list = []
    rmse_valid_list = []
    y_valid_pred_final = []
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.L1Loss()
    h_state = None

    epoch_start_time = time.time() #########

    for epoch in range(1, cfg.n_epochs + 1):
        mae_train = 0.0
        cnt = 0
        for start in range(len(x_train) - cfg.batch_size + 1):
            net.train()
            progress = start / (len(x_train) - cfg.batch_size + 1)

            x_input = torch.tensor(x_train[start:start + cfg.batch_size], dtype=torch.float32)
            y_true = torch.tensor(y_train[start:start + cfg.batch_size], dtype=torch.float32)

            if cfg.model_name == 'RNN' or cfg.model_name == 'GRU' or cfg.model_name == 'gMamba':
                y_pred, _h_state = net(x_input, h_state)
                h_state = _h_state.detach()
            else:
                y_pred = net(x_input)

            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # mse_train_batch = loss.data
            # rmse_train_batch = np.sqrt(mse_train_batch)
            # rmse_train += mse_train_batch ##############yuan

            mae_train_batch = loss.data
            mae_train += mae_train_batch ##############gai

            if start % int((len(x_train) - cfg.batch_size) / 5) == 0:
                print_time = time.time()
                print('epoch: {}  progress: {:.0f}%  loss: {:.3f}  mae: {:.3f}  time: {:.2f}s'.format(
                    epoch, progress * 100, loss, mae_train_batch, print_time - epoch_start_time))
            cnt += 1

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print("Time consumed for epoch {}: {:.2f} seconds".format(epoch, epoch_time))

        # 重置 epoch 开始时间
        epoch_start_time = time.time()

        # rmse_train = np.sqrt(rmse_train / cnt) yuan
        mae_train = mae_train / cnt #gai


        # validation
        net.eval()
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
            # rmse_valid_batch = np.sqrt(mse_valid_batch) yuan
            # me_valid_batch = mae_valid_batch# gai

            mae_valid += mae_valid_batch
            cnt += 1
        y_valid_pred_final = np.array(y_valid_pred_final).reshape((-1, 1))
        # rmse_valid = np.sqrt(rmse_valid / cnt) #yuan
        mae_valid = mae_valid / cnt #gai
        rmse_valid = np.sqrt(metrics.mean_squared_error(y_valid, y_valid_pred_final))

        mae_train_list.append(mae_train)
        mae_valid_list.append(mae_valid)
        rmse_valid_list.append(rmse_valid)

        # save the best model
        if mae_valid == np.min(mae_valid_list):
            torch.save(net.state_dict(), cfg.model_save_pth)

        print('\n>>> epoch: {}  MAE_train: {:.4f}  RMSE_valid: {:.4f} MAE_valid: {:.4f}\n'
              '    RMSE_valid_min: {:.4f}  MAE_valid_min: {:.4f}\n'
              .format(epoch, mae_train, rmse_valid, mae_valid, np.min(rmse_valid_list), np.min(mae_valid_list)))


def main():
    # Hyper Parameters
    cfg.print_params()
    np.random.seed(cfg.rand_seed)
    torch.manual_seed(cfg.rand_seed)

    # Load data
    print('\nLoading data...\n')
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y)
    #
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
    elif cfg.model_name == 'Mamba':
        net = models.SimpleMamba(d_model=cfg.d_model,d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                            d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
    elif cfg.model_name == 'oMamba':
        net = models.oMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, out_in_channels=cfg.out_in_channels,
                            d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                            d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
    elif cfg.model_name == 'STCN':
        net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                          num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)

    elif cfg.model_name == 'gMamba':
        net = models.gMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, out_in_channels=cfg.out_in_channels, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                            d_inner = cfg.d_inner, dt_rank = cfg.dt_rank)
        # net = models.gMamba(input_size=cfg.input_size, in_channels=cfg.in_channels, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
        #                     d_inner=cfg.d_inner, dt_rank=cfg.dt_rank)
        # net = models.gMamba(input_size=cfg.input_size, in_channels=cfg.in_channels,
        #                     output_size=cfg.output_size,
        #                     num_channels=[cfg.hidden_size] * cfg.levels, kernel_size=cfg.kernel_size,
        #                     dropout=cfg.dropout)
      #

    print('\n------------ Model structure ------------\nmodel name: {}\n{}\n-----------------------------------------\n'.format(cfg.model_name, net))
    # sys.exit()
    # Training
    print('\nStart training...\n')
    train(net, x_train, y_train, x_valid, y_valid, x_test, y_test)


if __name__ == '__main__':
    main()
