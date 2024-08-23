import numpy as np
import torch
import scipy.io as scio
import torch
import torch.nn as nn

# 选择实验数据集
problem = 'Burgers' # 'Burgers', 'chafee-infante', 'Kdv', 'PDE_divide', 'PDE_compound', 'advection', 'ReacDiff', 'heat'
seed = 0
device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

###########################################################################################
# Neural network
max_epoch = 100 * 1000
path = 'model/'+problem+'_sine_sin_50_3fc2_'+'%d'%(max_epoch/1000)+'k_Adam.pkl'
hidden_dim = 50

train_ratio = 1  # the ratio of training dataset
num_feature = 2  # 该算法使用的数据集都是形如u(x,t)的，因此特征数为2。如果要使用自己的数据集，需要修改为7，即u(x1,x2,x3,x4,x5,x6,t)
normal = True

###########################################################################################
# Metadata
fine_ratio = 2 # 通过MetaData加密数据的倍数
use_metadata = False
delete_edges = False
print('use_metadata =', use_metadata)
print('delete_edges =', delete_edges)

# AIC hyperparameter
aic_ratio = 1  # lower this ratio, less important is the number of elements to AIC value


print(path)
print(device)
print('fine_ratio = ',fine_ratio)
###########################################################################################
# 这个网络是在干啥？为什么前馈操作中还有sin？
class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(int(n_hidden),n_output)
    def forward(self,x):
        out = torch.sin((self.fc1(x)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = self.predict(out) 
        return out

# Data
def divide(up, down, eta=1e-10):
    while np.any(down == 0):
        down += eta
    return up/down

# PDE_divide: ut = -ux/x + 0.25*uxx
if problem == 'PDE_divide':
    u=np.load("./data/PDE_divide.npy").T  # (100,251)
    nx = 100
    nt = 251
    x=np.linspace(1,2,nx)
    t=np.linspace(0,1,nt)
    right_side = 'right_side = -config.divide(ux, x) + 0.25*uxx'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin'
    left_side_origin = 'left_side_origin = ut_origin'

# PDE_compound: ut = d(uux)(x) = u*uxx + ux*ux
if problem == 'PDE_compound':
    u=np.load("./data/PDE_compound.npy").T  # (100,251)
    nx = 100
    nt = 251
    x=np.linspace(1,2,nx)
    t=np.linspace(0,0.5,nt)
    right_side = 'right_side = u*uxx + ux*ux'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = u_origin*uxx_origin + ux_origin*ux_origin'
    left_side_origin = 'left_side_origin = ut_origin'
    
# Burgers: ut = -u*ux + 0.1*uxx
if problem == 'Burgers':
    data = scio.loadmat('./data/burgers.mat')  # (256,201)
    u=data.get("usol")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("t").reshape(1,201))
    right_side = 'right_side = -u*ux+0.1*uxx'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
    left_side_origin = 'left_side_origin = ut_origin'

# Kdv: ut = -0.0025*uxxx - uux
if problem == 'Kdv':
    data = scio.loadmat('./data/Kdv.mat')  # (512,201)
    u=data.get("uu")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("tt").reshape(1,201))
    right_side = 'right_side = -0.0025*uxxx-u*ux'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
    left_side_origin = 'left_side_origin = ut_origin'

# chafee-infante: ut = u_xx - u + u**3
if problem == 'chafee-infante':
    u = np.load("./data/chafee_infante_CI.npy")  # (x,t) (301,200)
    x = np.load("./data/chafee_infante_x.npy")
    t = np.load("./data/chafee_infante_t.npy") 
    right_side = 'right_side = uxx-u+u**3'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = uxx_origin-u_origin+u_origin**3'
    left_side_origin = 'left_side_origin = ut_origin'

# advection: ut = -ux
if problem == 'advection':
    u = np.load("./data/advection_u.npy").T  # (x,t) (1024,201)
    x = np.load("./data/advection_x.npy")
    t = np.load("./data/advection_t.npy")
    right_side = 'right_side = -ux'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -ux_origin'
    left_side_origin = 'left_side_origin = ut_origin'

# ReacDiff: ut = uxx + u - u**2
if problem == 'ReacDiff':
    u = np.load("./data/ReacDiff_u.npy").T  # (x,t) (1024,101)
    x = np.load("./data/ReacDiff_x.npy")
    t = np.load("./data/ReacDiff_t.npy")
    right_side = 'right_side = uxx + u - u**2'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = uxx_origin+u_origin*(1-u_origin)'
    left_side_origin = 'left_side_origin = ut_origin'

# 数据集有问题，shape不是(x,y,t)，不知道怎么改。得在heat_deepxde中画图的部分改data数组？
# heat: ut = 0.5*uxx + 0.5uyy
if problem == 'heat':
    u = np.load("./data/heat_u.npy")  # (<x,y>,t) (40000,21)
    x = np.load("./data/heat_x.npy")  # (40000,)
    y = np.load("./data/heat_y.npy")  # (40000,)
    t = np.load("./data/heat_t.npy")  # (21,)
    right_side = 'right_side = 0.5*uxx'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = 0.5*uxx_origin'
    left_side_origin = 'left_side_origin = ut_origin'


# 超参数设定
# problem = 'chaffee-infante' # choose the dataset

# seed = 0 # set the random seed

# fine_ratio = 2 # the ratio of Metadata set to original dataset. A ratio of 2 means that the sampling interval of Metadata is twice that of original data.

# use_metadata = False # whether to use Metadata

# delete_edges = False # whether to delete the Metadata on the boundaries of the field where the derivatives are not accurate based on finite difference.

# aic_ratio = 1  # the hyperparameter in the AIC. lower this ratio, less important is the number of elements to AIC value.

# max_epoch = 100 * 1000 # Hyperparameter to generate the Metadata Neural Networks.

# hidden_dim = 50 # Hyperparameter to generate the Metadata Neural Networks. Number of neurons in the hidden layer.

# normal = True # Hyperparameter to generate the Metadata Neural Networks. Whether to normalize the inputs.