# coding=utf-8

# model hyper-parameters
rand_seed = 314 #314 350
f_x = './data/xy/x_1013.pkl'
f_y = './data/xy/y_1013.pkl'

# f_x = ['./data/xy/x_1013_spring.pkl', './data/xy/x_1013_summer.pkl',
#             './data/xy/x_1013_autumn.pkl', './data/xy/x_1013_winter.pkl']
# f_y = ['./data/xy/y_1013_spring.pkl', './data/xy/y_1013_summer.pkl',
#             './data/xy/y_1013_autumn.pkl', './data/xy/y_1013_winter.pkl']
model_name = ('STCN')  # ['RNN', 'GRU', 'LSTM', 'TCN', 'STCN','Mamba', 'oMamba', 'gMamba']
device = 'cuda'  # 'cpu' or 'cuda'
input_size = 12 # 12
hidden_size = 32 # ########## 32
output_size = 1
num_layers = 4 #### 4
levels = 4 #4
kernel_size = 4  ###############4
dropout = 0.25 ##########0.25
in_channels = 18          ###city
[198, 0.001269, 12, 3, 4, 24, 115]
[32, 0.003665, 15, 3, 2, 119, 33]
out_in_channels =230   ####64
d_model=12
##########
d_state = 27   #16       ..32
d_conv = 3      #4        ..3
expand = 10       #2       ..4
d_inner = 158    #expend*dmodel  ..12
dt_rank = 45      #dmodel/16
##
# hidden_size = 32


batch_size = 1
lr =0.001747   #1e-3
n_epochs = 50
model_save_pth = './models/model_{}.pth'.format(model_name)


def print_params():
    print('\n------ Parameters ------')
    print('rand_seed = {}'.format(rand_seed))
    print('f_x = {}'.format(f_x))
    print('f_y = {}'.format(f_y))
    print('device = {}'.format(device))
    # print('input_size = {}'.format(input_size))
    # print('hidden_size = {}'.format(hidden_size))
    # print('num_layers = {}'.format(num_layers))
    # print('output_size = {}'.format(output_size))
    # print('levels (for TCN) = {}'.format(levels))
    # print('kernel_size (for TCN) = {}'.format(kernel_size))
    # print('dropout (for TCN) = {}'.format(dropout))
    # print('in_channels (for STCN) = {}'.format(in_channels))
    # print('batch_size = {}'.format(batch_size))
    # print('lr = {}'.format(lr))
    # print('n_epochs = {}'.format(n_epochs))
    print('model_save_pth = {}'.format(model_save_pth))
    print('------------------------\n')
