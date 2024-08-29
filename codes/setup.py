import numpy as np
from PDE_find import Diff, Diff2, FiniteDiff
import Data_generator as Data_generator
from requests import get
import random
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import configure as config
from configure import divide

# 1.从Data_generator加载数据
# 2.评估偏微分方程与观测值之间的适应度（计算偏微分方程左侧和右侧之间的误差）。偏微分方程中涉及的导数可以通过有限差分或自动微分来计算
# 3.绘制不同阶数导数的图形，给定偏微分方程的左右两侧
# 4.在 SGA 中设置运算符和操作数
simple_mode = True
see_tree = None
plot_the_figures = False
use_metadata = False 
use_difference = False
use_extend = True

def cubic(inputs):
        return np.power(inputs, 3)

if use_extend == False:
    if use_difference == True:
        use_autograd = False
        print('Using difference method')
    else:
        use_autograd = True
        print('Using autograd method')

    def get_random_int(max_int):
        random_result = get('https://www.random.org/integers/?num=1&min=0&max={0}&col=1&base=10&format=plain&rnd=new'.format(max_int)).content
        try:
            int(random_result)
        except:
            print(random_result)
        return int(random_result)

    # rand = get_random_int(1e6)
    rand = config.seed #0
    print('random seed: {}'.format(rand))
    # 237204
    np.random.seed(rand)
    random.seed(rand)

    # 1.从Data_generator加载数据
    # load Metadata
    u = Data_generator.u
    x = Data_generator.x
    t = Data_generator.t
    x_all = Data_generator.x_all
    n, m = u.shape
    dx = x[2]-x[1]
    dt = t[1]-t[0]
    # 扩充维度使得x和t与u的size相同
    x = np.tile(x, (m, 1)).transpose((1, 0))  # (n,m) (256,201)
    x_all = np.tile(x_all, (m, 1)).transpose((1, 0))  # (n,m) (256,201)
    t = np.tile(t, (n, 1))  # (n,m) (256,201)

    # load Origin data
    u_origin=config.u
    x_origin=config.x
    t_origin=config.t
    n_origin, m_origin = u_origin.shape
    dx_origin = x_origin[2]-x_origin[1]
    dt_origin = t_origin[1]-t_origin[0]
    # 扩充维度使得与u的size相同
    x_origin = np.tile(x_origin, (m_origin, 1)).transpose((1, 0))
    t_origin = np.tile(t_origin, (n_origin, 1))

    # 差分
    # calculate the error of correct cofs & correct terms
    if use_difference == True:
        ut = np.zeros((n, m))
        for idx in range(n):
            ut[idx, :] = FiniteDiff(u[idx, :], dt)
        ux = np.zeros((n, m))
        uxx = np.zeros((n, m))
        uxxx = np.zeros((n, m))
        for idx in range(m):
            ux[:, idx] = FiniteDiff(u[:, idx], dx)  #idx is the id of one time step
        for idx in range(m):
            uxx[:, idx] = FiniteDiff(ux[:, idx], dx)
        for idx in range(m):
            uxxx[:, idx] = FiniteDiff(uxx[:, idx], dx)

        ut_origin = np.zeros((n_origin, m_origin))
        for idx in range(n_origin):
            ut_origin[idx, :] = FiniteDiff(u_origin[idx, :], dt_origin)
        ux_origin = np.zeros((n_origin, m_origin))
        uxx_origin = np.zeros((n_origin, m_origin))
        uxxx_origin = np.zeros((n_origin, m_origin))
        for idx in range(m_origin):
            ux_origin[:, idx] = FiniteDiff(u_origin[:, idx], dx_origin)  #idx is the id of one time step
        for idx in range(m_origin):
            uxx_origin[:, idx] = FiniteDiff(ux_origin[:, idx], dx_origin)
        for idx in range(m_origin):
            uxxx_origin[:, idx] = FiniteDiff(uxx_origin[:, idx], dx_origin)

    # autograd 问题在于被求导的部分形式不确定，如果每次重新训练神经网络，代价过高。
    if use_autograd == True:
        # load model
        hidden_dim = config.hidden_dim
        num_feature = config.num_feature
        model = config.Net(num_feature, hidden_dim, 1)
        model.load_state_dict(torch.load(config.path))
        # autograd
        def fun(x, t, Net):
            database = torch.cat((x,t), 1)
            database = Variable(database, requires_grad=True)
            PINNstatic = Net(database.float())
            H_grad = torch.autograd.grad(outputs=PINNstatic.sum(), inputs=database, create_graph=True)[0]
            Ht = H_grad[:, 1]
            Hx = H_grad[:, 0]
            Ht_n = Ht.data.numpy()
            Hx_n=Hx.data.numpy()
            Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, 0]
            Hxx_n = Hxx.data.numpy()
            Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, 0]
            Hxxx_n = Hxxx.data.numpy()
            Htt = torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, 1]
            Htt_n = Htt.data.numpy()
            return Hx_n, Hxx_n, Hxxx_n, Ht_n
        x_1d = np.reshape(x, (n*m, 1))
        t_1d = np.reshape(t, (n*m, 1))
        ux, uxx, uxxx, ut = fun(torch.from_numpy(x_1d), torch.from_numpy(t_1d), model)
        ut = np.reshape( ut, (n,m))
        ux = np.reshape( ux, (n,m))
        uxx = np.reshape( uxx, (n,m))
        uxxx = np.reshape( uxxx, (n,m))

        x_1d_origin = np.reshape(x_origin, (n_origin*m_origin, 1))
        t_1d_origin = np.reshape(t_origin, (n_origin*m_origin, 1))
        ux_origin, uxx_origin, uxxx_origin, ut_origin = fun(torch.from_numpy(x_1d_origin), torch.from_numpy(t_1d_origin), model)
        ut_origin = np.reshape( ut_origin, (n_origin,m_origin))
        ux_origin = np.reshape( ux_origin, (n_origin,m_origin))
        uxx_origin = np.reshape( uxx_origin, (n_origin,m_origin))
        uxxx_origin = np.reshape( uxxx_origin, (n_origin,m_origin))

    # 2.评估偏微分方程与观测值之间的适应度（计算偏微分方程左侧和右侧之间的误差）。偏微分方程中涉及的导数可以通过有限差分或自动微分来计算
    # calculate error
    # config.right_side为一个字符串，exec是执行这个字符串，目的是固定方程的左右两侧
    # 如果use_metadata = True，那么error和error_origin相同
    exec (config.right_side)  # right_side = -config.divide(ux, x) + 0.25*uxx
    exec (config.left_side)  # left_side = ut
    n1, n2, m1, m2 = int(n*0.1), int(n*0.9), int(m*0), int(m*1)
    right_side_full = right_side
    right_side = right_side[n1:n2, m1:m2]
    left_side = left_side[n1:n2, m1:m2]
    right = np.reshape(right_side, ((n2-n1)*(m2-m1), 1))
    left = np.reshape(left_side, ((n2-n1)*(m2-m1), 1))
    diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
    print('data error without edges',diff)

    exec (config.right_side_origin)  # right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin
    exec (config.left_side_origin)  # left_side_origin = ut_origin
    n1_origin, n2_origin, m1_origin, m2_origin = int(n_origin*0.1), int(n_origin*0.9), int(m_origin*0), int(m_origin*1)
    right_side_full_origin = right_side_origin
    right_side_origin = right_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
    left_side_origin = left_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
    right_origin = np.reshape(right_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
    left_origin = np.reshape(left_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
    diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((n2_origin-n1_origin)*(m2_origin-m1_origin))
    print('data error_origin without edges',diff_origin)

    exec (config.right_side)
    exec (config.left_side)
    n1, n2, m1, m2 = int(n*0), int(n*1), int(m*0), int(m*1)
    right_side_full = right_side
    right_side = right_side[n1:n2, m1:m2]
    left_side = left_side[n1:n2, m1:m2]
    right = np.reshape(right_side, ((n2-n1)*(m2-m1), 1))
    left = np.reshape(left_side, ((n2-n1)*(m2-m1), 1))
    diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
    print('data error',diff)

    exec (config.right_side_origin)
    exec (config.left_side_origin)
    n1_origin, n2_origin, m1_origin, m2_origin = int(n_origin*0), int(n_origin*1), int(m_origin*0), int(m_origin*1)
    right_side_full_origin = right_side_origin
    right_side_origin = right_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
    left_side_origin = left_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
    right_origin = np.reshape(right_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
    left_origin = np.reshape(left_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
    diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((n2_origin-n1_origin)*(m2_origin-m1_origin))
    print('data error_origin',diff_origin)

    # 3.绘制不同阶数导数的图形，给定偏微分方程的左右两侧
    # plot the figures
    if plot_the_figures == True:
        from matplotlib.pyplot import MultipleLocator
        path_prefix = 'output/setup/' + config.problem + '/'
        x1 = int(n_origin*0.1)
        x2 = int(n_origin*0.9)
        t1 = int(m_origin*0.1)
        t2 = int(m_origin*0.9)
        # Plot the flow field
        plt.figure(figsize=(10,3))
        mm1=plt.imshow(u, interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max(u_origin), vmin=np.min(u_origin))
        plt.colorbar().ax.tick_params(labelsize=16) 
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('Metadata Field', fontsize = 15)
        plt.savefig(path_prefix + config.problem + '_Metadata_field_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

        plt.figure(figsize=(10,3))
        mm1=plt.imshow(u_origin, interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max(u_origin), vmin=np.min(u_origin))
        plt.colorbar().ax.tick_params(labelsize=16) 
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('Original Field', fontsize = 15)
        plt.savefig(path_prefix + config.problem + '_Original_field_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

        # Plot the PDE terms
        fig=plt.figure(figsize=(5,3))
        ax = fig.add_subplot(1, 1, 1)
        x_index = np.linspace(0,256, n_origin)
        x_index_fine = np.linspace(0,100, n)
        if use_metadata == True:
            plt.plot(x_index_fine, ut[:,int(m/2)], color='red', label = 'Metadata')
        plt.plot(x_index, ut_origin[:,int(m_origin/2)], color='blue', linestyle='--') #, label = 'Raw data' # 中间时刻的ut
        # plt.ylim(np.min(ut_origin[x1:x2,t1:t2]), np.max(ut_origin[x1:x2,t1:t2]))
        # plt.title('$U_t$ (Left side)')
        ax.set_ylabel('$U_t$', fontsize=18)
        ax.set_xlabel('x', fontsize=18)
        # plt.legend(loc='upper left')
        x_major_locator=MultipleLocator(32)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(path_prefix + config.problem + '_Ut_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

        fig=plt.figure(figsize=(5,3))
        ax = fig.add_subplot(1, 1, 1)
        x_index = np.linspace(0,256, n_origin)
        x_index_fine = np.linspace(0,100, n)
        if use_metadata == True:
            plt.plot(x_index_fine, ux[:,int(m/2)], color='red', label = 'Metadata')
        plt.plot(x_index, ux_origin[:,int(m_origin/2)], color='blue', linestyle='--') #, label = 'Raw data'
        # plt.ylim(np.min(ux_origin[x1:x2,t1:t2]), np.max(ux_origin[x1:x2,t1:t2]))
        # plt.title('$U_x$')
        ax.set_ylabel('$U_x$', fontsize=18)
        ax.set_xlabel('x', fontsize=18)
        # plt.legend(loc='upper left')
        x_major_locator=MultipleLocator(32)
        ax=plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        plt.savefig(path_prefix + config.problem + '_Ux_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

        fig=plt.figure(figsize=(5,3))
        ax = fig.add_subplot(1, 1, 1)
        x_index = np.linspace(0,100, n_origin)
        x_index_fine = np.linspace(0,100, n)
        if use_metadata == True:
            plt.plot(x_index_fine, uxx[:,int(m/2)], color='red', label = 'Metadata')
        plt.plot(x_index, uxx_origin[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
        # plt.ylim(np.min(uxx_origin[x1:x2,t1:t2]), np.max(uxx_origin[x1:x2,t1:t2]))
        # plt.title('$U_x$'+'$_x$')
        ax.set_ylabel('$U_x$'+'$_x$', fontsize=18)
        ax.set_xlabel('x', fontsize=18)
        plt.legend(loc='upper left')
        plt.savefig(path_prefix + config.problem + '_Uxx_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

        plt.figure(figsize=(5,3))
        x_index = np.linspace(0,100, n_origin)
        x_index_fine = np.linspace(0,100, n)
        if use_metadata == True:
            plt.plot(x_index_fine, u[:,int(m/2)], color='red', label = 'Metadata')
        plt.plot(x_index, u_origin[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
        # plt.ylim(np.min(u_origin[x1:x2,t1:t2]), np.max(u_origin[x1:x2,t1:t2]))
        plt.title('U')
        plt.legend(loc='upper left')
        plt.savefig(path_prefix + config.problem + '_U_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

        plt.figure(figsize=(5,3))
        x_index = np.linspace(0,100, (n2_origin-n1_origin))
        x_index_fine = np.linspace(0,100, (n2-n1))
        if use_metadata == True:
            plt.plot(x_index_fine, right_side[:,int((m2-m1)/2)], color='red', label = 'Metadata')
        plt.plot(x_index, right_side_origin[:,int((m2_origin-m1_origin)/2)], color='blue', linestyle='--', label = 'Raw data')
        # plt.ylim(np.min(right_side_origin[x1:x2,t1:t2]), np.max(right_side_origin[x1:x2,t1:t2]))
        plt.title('Right side')
        plt.legend(loc='upper left')
        plt.savefig(path_prefix + config.problem + '_Right_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

        plt.figure(figsize=(5,3))
        x_index = np.linspace(0,100, n_origin)
        x_index_fine = np.linspace(0,100, n)
        if use_metadata == True:
            plt.plot(x_index_fine, (ut-right_side_full)[:,int(m/2)], color='red', label = 'Metadata')
        plt.plot(x_index, (ut_origin-right_side_full_origin)[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
        # plt.ylim(np.min(ut_origin[x1:x2,t1:t2]), np.max(ut_origin[x1:x2,t1:t2]))
        plt.title('Residual')
        plt.legend(loc='upper left')
        plt.savefig(path_prefix + config.problem + '_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

        plt.figure(figsize=(10,3))
        mm1=plt.imshow((ut-right_side_full), interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max((ut_origin-right_side_full_origin)), vmin=np.min((ut_origin-right_side_full_origin)))
        # mm1=plt.imshow((ut-right_side), interpolation='nearest',  cmap='Blues', origin='lower', vmax=5, vmin=-5)
        plt.colorbar().ax.tick_params(labelsize=16) 
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('Metadata Residual', fontsize = 15)
        plt.savefig(path_prefix + config.problem + '_Metadata_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

        plt.figure(figsize=(10,3))
        mm1=plt.imshow((ut_origin-right_side_full_origin), interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max((ut_origin-right_side_full_origin)), vmin=np.min((ut_origin-right_side_full_origin)))
        # mm1=plt.imshow((ut_origin-right_side_origin), interpolation='nearest',  cmap='Blues', origin='lower', vmax=5, vmin=-5)
        plt.colorbar().ax.tick_params(labelsize=16) 
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('Original Residual', fontsize = 15)
        plt.savefig(path_prefix + config.problem + '_Original_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    ###########################################################################################
    # for default evaluation
    # 把数据集中的数据重塑shape为(nxm,1) 
    default_u = np.reshape(u, (u.shape[0]*u.shape[1], 1))
    default_ux = np.reshape(ux, (u.shape[0]*u.shape[1], 1))
    default_uxx = np.reshape(uxx, (u.shape[0]*u.shape[1], 1))
    default_uxxx = np.reshape(uxxx, (u.shape[0]*u.shape[1], 1))
    default_u2 = np.reshape(u**2, (u.shape[0]*u.shape[1], 1))
    default_u3 = np.reshape(u**3, (u.shape[0]*u.shape[1], 1))
    # 设置默认项，需要按情况修改
    default_terms = np.hstack((default_u, default_ux, default_uxx, default_uxxx, default_u2, default_u3))
    default_names = ['u', 'ux', 'uxx', 'uxxx', 'u^2', 'u^3']
    # default_terms = np.hstack((default_u, default_ux, default_uxx, default_u2))
    # default_names = ['u', 'ux', 'uxx', 'u^2']
    # default_terms = np.hstack((default_u, default_ux))
    # default_names = ['u', 'ux']
    # default_terms = np.hstack((default_u)).reshape(-1,1)
    # default_names = ['u']

    zeros = np.zeros(u.shape)

    print("默认项shape:", default_terms.shape)
    num_default = default_terms.shape[1]  # 包含的默认项的数量，num_default中为一定包含的候选集
    print("默认项数量num_default:", num_default)


    if simple_mode:
        ALL = np.array([['+', 2, np.add], ['-', 2, np.subtract],['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
                        ['u', 0, u], ['x', 0, x], ['ux', 0, ux],  ['0', 0, zeros],
                        ['^2', 1, np.square], ['^3', 1, cubic]], dtype=object) #  ['u^2', 0, u**2], ['uxx', 0, uxx], ['t', 0, t],
        OPS = np.array([['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2], ['^2', 1, np.square], ['^3', 1, cubic]], dtype=object)
        ROOT = np.array([['*', 2, np.multiply], ['d', 2, Diff], ['d^2', 2, Diff2], ['/', 2, divide], ['^2', 1, np.square], ['^3', 1, cubic]], dtype=object)  # 根节点不包含+,-
        OP1 = np.array([['^2', 1, np.square], ['^3', 1, cubic]], dtype=object)  # 不能更改，否则sga程序运行会卡住
        OP2 = np.array([['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2]], dtype=object)
        # VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux], ['uxx', 0, uxx], ['u^2', 0, u**2]], dtype=object)  # 变量，按照情况进行修改，在PDE_compound数据集中起作用
        VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux]], dtype=object)
        den = np.array([['x', 0, x]], dtype=object)

    pde_lib, err_lib = [], []
else:
    if use_difference == True:
        use_autograd = False
        print('Using difference method')
    else:
        use_autograd = True
        print('Using autograd method')
    
    rand = config.seed
    print('random seed: {}'.format(rand))
    np.random.seed(rand)
    random.seed(rand)

    # u = np.load('data/heat_xyt.npy')  # (200,200,21) (n,m,k)
    # x = np.load('data/heat_x.npy')  # (200,) (n,)
    # y = np.load('data/heat_y.npy')  # (200,) (m,)
    # t = np.load('data/heat_t.npy')  # (21,) (k,)

    # u = np.load('data/heat_xyt_less.npy')  # (100,100,21) (n,m,k)
    # x = np.load('data/heat_x_less.npy')  # (100,) (n,)
    # y = np.load('data/heat_y_less.npy')  # (100,) (m,)
    # t = np.load('data/heat_t.npy')  # (21,) (k,)

    # u = np.load('data/heat_xyt_less_initup1.npy')  # (100,100,21) (n,m,k)
    # x = np.load('data/heat_x_less_initup1.npy')  # (100,) (n,)
    # y = np.load('data/heat_y_less_initup1.npy')  # (100,) (m,)
    # t = np.load('data/heat_t.npy')  # (21,) (k,)

    u = np.load('data/heat_xyt_less_initup1down1.npy')  # (100,100,21) (n,m,k)
    x = np.load('data/heat_x_less_initup1down1.npy')  # (100,) (n,)
    y = np.load('data/heat_y_less_initup1down1.npy')  # (100,) (m,)
    t = np.load('data/heat_t.npy')  # (21,) (k,)

    dx = x[2] - x[1]
    dy = y[2] - y[1]
    dt = t[1] - t[0]
    n, m, k = u.shape

    x = np.tile(x, (m, k, 1)).transpose((2, 0, 1))  # (m,k,n)->(n,m,k)
    y = np.tile(y, (n, k, 1)).transpose((0, 2, 1))  # (n,k,m)->(n,m,k)
    t = np.tile(t, (n, m, 1))  # (n,m,k)

    if use_difference == True:
        ut = Diff(u, dt, 't')
        ux = Diff(u, dx, 'x')
        uy = Diff(u, dy, 'y')
        uxx = Diff2(u, dx, 'x')
        uyy = Diff2(u, dy, 'y')
        
        default_u = np.reshape(u, (n*m*k, 1))
        default_ux = np.reshape(ux, (n*m*k, 1))
        default_uy = np.reshape(uy, (n*m*k, 1))
        default_uxx = np.reshape(uxx, (n*m*k, 1))
        default_uyy = np.reshape(uyy, (n*m*k, 1))
        
    if use_autograd == True:
        # u = np.load('data/heat_xyt.npy')  # (200,200,21)
        # ut = np.load('data/heat_ut.npy')  # (840000,1)
        # default_u = np.load('data/heat_out.npy')  # (840000,1)
        # default_ux = np.load('data/heat_ux.npy')  # (840000,1)
        # default_uy = np.load('data/heat_uy.npy')  # (840000,1)
        # default_uxx = np.load('data/heat_uxx.npy')  # (840000,1)
        # default_uyy = np.load('data/heat_uyy.npy')  # (840000,1)

        # u = np.load('data/heat_xyt_less.npy')  # (100,100,21)
        # ut = np.load('data/heat_ut_less.npy')  # (210000,1)
        # default_u = np.load('data/heat_out_less.npy')  # (210000,1)
        # default_ux = np.load('data/heat_ux_less.npy')  # (210000,1)
        # default_uy = np.load('data/heat_uy_less.npy')  # (210000,1)
        # default_uxx = np.load('data/heat_uxx_less.npy')  # (210000,1)
        # default_uyy = np.load('data/heat_uyy_less.npy')  # (210000,1)

        # u = np.load('data/heat_xyt_less_initup1.npy')  # (100,100,21)
        # ut = np.load('data/heat_ut_less_initup1.npy')  # (210000,1)
        # default_u = np.load('data/heat_out_less_initup1.npy')  # (210000,1)
        # default_ux = np.load('data/heat_ux_less_initup1.npy')  # (210000,1)
        # default_uy = np.load('data/heat_uy_less_initup1.npy')  # (210000,1)
        # default_uxx = np.load('data/heat_uxx_less_initup1.npy')  # (210000,1)
        # default_uyy = np.load('data/heat_uyy_less_initup1.npy')  # (210000,1)

        u = np.load('data/heat_xyt_less_initup1down1.npy')  # (100,100,21)
        ut = np.load('data/heat_ut_less_initup1down1.npy')  # (210000,1)
        default_u = np.load('data/heat_out_less_initup1down1.npy')  # (210000,1)
        default_ux = np.load('data/heat_ux_less_initup1down1.npy')  # (210000,1)
        default_uy = np.load('data/heat_uy_less_initup1down1.npy')  # (210000,1)
        default_uxx = np.load('data/heat_uxx_less_initup1down1.npy')  # (210000,1)
        default_uyy = np.load('data/heat_uyy_less_initup1down1.npy')  # (210000,1)
        
        ux = np.reshape(default_ux, (n, m, k))
        uy = np.reshape(default_uy, (n, m, k))
        uxx = np.reshape(default_uxx, (n, m, k))
        uyy = np.reshape(default_uyy, (n, m, k))
        ut = np.reshape(ut, (n, m, k))

    exec (config.right_side)
    exec (config.left_side)
    n1, n2, m1, m2, k1, k2 = int(n*0), int(n*1), int(m*0), int(m*1), int(k*0), int(k*1)
    right_side = right_side[n1:n2, m1:m2, k1:k2]
    left_side = left_side[n1:n2, m1:m2, k1:k2]
    right = np.reshape(right_side, ((n2-n1)*(m2-m1)*(k2-k1), 1))
    left = np.reshape(left_side, ((n2-n1)*(m2-m1)*(k2-k1), 1))
    diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1)*(k2-k1))
    print('data error', diff)  # 用差分计算的导数，diff=0.04；用外部的导数，diff=0.02

    exec (config.right_side)
    exec (config.left_side)
    n1, n2, m1, m2, k1, k2 = int(n*0.1), int(n*0.9), int(m*0.1), int(m*0.9), int(k*0), int(k*1)
    right_side = right_side[n1:n2, m1:m2, k1:k2]
    left_side = left_side[n1:n2, m1:m2, k1:k2]
    right = np.reshape(right_side, ((n2-n1)*(m2-m1)*(k2-k1), 1))
    left = np.reshape(left_side, ((n2-n1)*(m2-m1)*(k2-k1), 1))
    diff_without_edges = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1)*(k2-k1))
    print('data error without edges', diff_without_edges)  # 用差分计算的导数，diff_without_edges=0.01；用外部的导数，diff_without_edges=0.002

    # 设置默认项，需要按情况修改
    default_terms = np.hstack((default_u, default_ux, default_uy, default_uxx, default_uyy))
    default_names = ['u', 'ux', 'uy', 'uxx', 'uyy']
    # default_terms = np.hstack((default_uxx, default_uyy))
    # default_names = ['uxx', 'uyy']

    print("默认项shape:", default_terms.shape)
    num_default = default_terms.shape[1]  # 包含的默认项的数量，num_default中为一定包含的候选集
    print("默认项数量num_default:", num_default)

    zeros = np.zeros(u.shape)

    if simple_mode:
        ALL = np.array([['+', 2, np.add], ['-', 2, np.subtract],['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
                        ['u', 0, u], ['x', 0, x], ['ux', 0, ux],  ['0', 0, zeros],
                        ['^2', 1, np.square], ['^3', 1, cubic], ['y', 0, y], ['uy', 0, uy]], dtype=object)
        OPS = np.array([['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2], ['^2', 1, np.square], ['^3', 1, cubic]], dtype=object)
        ROOT = np.array([['*', 2, np.multiply], ['d', 2, Diff], ['d^2', 2, Diff2], ['/', 2, divide], ['^2', 1, np.square], ['^3', 1, cubic]], dtype=object)  # 根节点不包含+,-
        OP1 = np.array([['^2', 1, np.square], ['^3', 1, cubic]], dtype=object)  # 不能更改，否则sga程序运行会卡住
        OP2 = np.array([['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2]], dtype=object)
        # VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux], ['uxx', 0, uxx], ['u^2', 0, u**2]], dtype=object)  # 变量，按照情况进行修改，在PDE_compound数据集中起作用
        VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux], ['y', 0, y], ['uy', 0, uy]], dtype=object)
        # VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux], ['y', 0, y], ['uy', 0, uy], ['uxx', 0, uxx], ['uyy', 0, uyy]], dtype=object)
        den = np.array([['x', 0, x], ['y', 0, y]], dtype=object)

    pde_lib, err_lib = [], []

# 超参数设定
# simple_mode = True  # use the simple operators and operands.

# use_metadata = False  # whether to use the Metadata.

# use_difference = True # whether to use finite difference or autograd to calculate the gradients.