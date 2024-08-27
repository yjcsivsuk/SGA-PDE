import numpy as np
import torch
import Data_generator as Data_generator
import configure as config

# 采用有限差分法计算函数项中的导数。使用 STRidge 查找当前迭代步骤中所有函数项（树）的最优组合，并评估当前迭代步骤的最优组合（即发现的偏微分方程）与观测值之间的拟合度
seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

u = Data_generator.u
x = Data_generator.x
t = Data_generator.t

# 一阶有限差分
def FiniteDiff(u, dx):
    """
    Calculate the first-order finite difference of a 1D array u.
    :param u: 1D numpy array
    :param dx: Grid spacing
    :return: 1D numpy array containing the first-order differences
    """
    n = u.size
    ux = np.zeros(n)
    for i in range(1, n - 1):
        ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)
    ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
    ux[n - 1] = (3.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
    return ux

# 二阶有限差分
def FiniteDiff2(u, dx):
    """
    Calculate the second-order finite difference of a 1D array u.
    :param u: 1D numpy array
    :param dx: Grid spacing
    :return: 1D numpy array containing the second-order differences
    """
    n = u.size
    ux = np.zeros(n)
    for i in range(1, n - 1):
        ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2
    ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
    ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
    return ux

# 求一阶导
def Diff(u, dxt, name):
    """
    Calculate the first-order difference of a 2D/3D array u along a specified dimension.
    :param u: 2D/3D numpy array
    :param dxt: Grid spacing
    :param name: Dimension along which to compute the difference ('x', 'y', or 't')
    :return: 2D/3D numpy array containing the first-order differences
    """
    if len(u.shape) == 2:
        n, m = u.shape
        uxt = np.zeros((n, m))
        if name == 'x':
            for i in range(m):
                uxt[:, i] = FiniteDiff(u[:, i], dxt)
        elif name == 't':
            for i in range(n):
                uxt[i, :] = FiniteDiff(u[i, :], dxt)
        else:
            NotImplementedError()
    elif len(u.shape) == 3:
        uxt = np.zeros_like(u)
        if name == 'x':
            for i in range(u.shape[1]):
                for j in range(u.shape[2]):
                    uxt[:, i, j] = FiniteDiff(u[:, i, j], dxt)
        elif name == 'y':
            for i in range(u.shape[0]):
                for j in range(u.shape[2]):
                    uxt[i, :, j] = FiniteDiff(u[i, :, j], dxt)
        elif name == 't':
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    uxt[i, j, :] = FiniteDiff(u[i, j, :], dxt)
        else:
            raise ValueError("Name must be 'x', 'y', or 't'")
    return uxt

# 求二阶导
def Diff2(u, dxt, name):
    """
    Calculate the first-order difference of a 2D/3D array u along a specified dimension.
    :param u: 2D/3D numpy array
    :param dxt: Grid spacing
    :param name: Dimension along which to compute the difference ('x', 'y', or 't')
    :return: 2D/3D numpy array containing the first-order differences
    """
    if len(u.shape) == 2:
        n, m = u.shape
        uxt = np.zeros((n, m))
        if name == 'x':
            for i in range(m):
                uxt[:, i] = FiniteDiff2(u[:, i], dxt)
        elif name == 't':
            for i in range(n):
                uxt[i, :] = FiniteDiff2(u[i, :], dxt)
        else:
            NotImplementedError()
    elif len(u.shape) == 3:
        uxt = np.zeros_like(u)
        if name == 'x':
            for i in range(u.shape[1]):
                for j in range(u.shape[2]):
                    uxt[:, i, j] = FiniteDiff2(u[:, i, j], dxt)
        elif name == 'y':
            for i in range(u.shape[0]):
                for j in range(u.shape[2]):
                    uxt[i, :, j] = FiniteDiff2(u[i, :, j], dxt)
        elif name == 't':
            for i in range(u.shape[0]):
                for j in range(u.shape[1]):
                    uxt[i, j, :] = FiniteDiff2(u[i, j, :], dxt)
        else:
            raise ValueError("Name must be 'x', 'y', or 't'")
    return uxt

# train函数使用STRidge训练一个预测器。它在不同的tol值上运行，并在训练集上训练预测器，然后在保留集上使用损失函数对它们进行评估。
def Train(R, Ut, lam, d_tol, AIC_ratio=1, maxit=10, STR_iters=10, l0_penalty=1, normalize=2, split=0.8,
          print_best_tol=False, sparse='STR'):
    """
    This function trains a predictor using STRidge.
    It runs over different values of tolerance and trains predictors on a training set, then evaluates them
    using a loss function on a holdout set.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    # np.random.seed(0)  # for consistancy
    n, _ = R.shape

    #train = np.random.choice(n, int(n * split), replace=False)
    #test = [i for i in np.arange(n) if i not in train]
    TrainR = R  # [train, :]
    TestR = R  # [test, :]
    TrainY = Ut  # [train, :]
    TestY = Ut  # [test, :]
    D = TrainR.shape[1]

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001 * np.linalg.cond(R)

    # check
    # print(np.nan in TrainR)
    # print(np.inf in TrainR)

    def AIC(w, err): # 输入当前error和权重w，给出AIC值
        k = 0
        for item in w:
            if abs(item) != 0:
                k += 1
        return AIC_ratio*2*k + 2*np.log(err)

    w_best = np.linalg.lstsq(TrainR, TrainY)[0]
    # 基于w_best计算误差，包括data_error_best，AIC_best
    data_err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2)
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty * np.count_nonzero(w_best)
    mse_best = np.mean((TrainY-TrainR.dot(w_best))**2)
    aic_best = AIC(w_best[:, 0], mse_best)
    tol_best = 0

    if sparse == 'STR':
        # Now increase tolerance until test performance decreases
        for iter in range(maxit):
            # Get a set of coefficients and error
            w = STRidge(R, Ut, lam, STR_iters, tol, normalize=normalize)
            # err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty * np.count_nonzero(w)
            data_err = np.linalg.norm(TestY - TestR.dot(w), 2)
            mse = np.mean((TrainY - TrainR.dot(w)) ** 2)

            # Has the accuracy improved?
            aic = AIC(w[:, 0], mse)
            if aic <= aic_best:
                aic_best = aic
                # err_best = err
                mse_best = mse
                w_best = w
                data_err_best = data_err
                tol_best = tol
                tol = tol + d_tol
            else:
                tol = max([0, tol - 2 * d_tol])
                d_tol = 2 * d_tol / (maxit - iter)
                tol = tol + d_tol

        if print_best_tol: print("Optimal tolerance:", tol_best)

    elif sparse == 'Lasso':
        w = Lasso(R, Ut, lam, w=np.array([0]), maxit=maxit*10, normalize=normalize)
        # err = np.linalg.norm(Ut - R.dot(w), 2) + l0_penalty * np.count_nonzero(w)
        data_err_best = np.linalg.norm(Ut - R.dot(w), 2)
        mse_best = np.mean((TrainY - TrainR.dot(w)) ** 2)
        aic_best = AIC(w[:, 0], mse_best)

    return w_best, data_err_best, mse_best, aic_best


def STRidge(X0, y, lam, maxit, tol, normalize=0, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.
    This assumes y is only one column
    """
    n, d = X0.shape
    X = np.zeros((n, d))
    # First normalize data
    # 根据情况是否标准化矩阵
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0
    # Get the standard ridge estimate
    # 如果lam不为0，则添加正则化项
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y))[0]
    else:
        w = np.linalg.lstsq(X, y)[0]
    num_relevant = d  # 当前重要的特征数量
    biginds = np.where(abs(w) > tol)[0]  # 权重系数绝对值大于 tol 的特征索引
    # Threshold and continue
    # 通过迭代减少不重要特征的权重，直到没有更多的特征被剔除或达到最大迭代次数
    for j in range(maxit):
        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]  # 权重系数绝对值小于 tol 的特征索引
        new_biginds = [i for i in range(d) if i not in smallinds]
        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                break
        biginds = new_biginds
        # Otherwise get a new guess
        w[smallinds] = 0  # 将权重系数绝对值小于 tol 的特征系数置0
        # 在置完0后，还需要重新计算剩下的权重
        if lam != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y))[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    # Now that we have the sparsity pattern, use standard least squares to get w
    # 在迭代完成后，使用标准最小二乘法重新计算重要特征的权重
    if biginds != []: 
        w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]
    # 如果数据进行了标准化，则返回的权重系数需要乘以标准化系数 Mreg 。否则，直接返回计算得到的权重系数 w
    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w


def Lasso(X0, Y, lam, w=np.array([0]), maxit=100, normalize=2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1
    """
    # Obtain size of X
    n, d = X0.shape
    X = np.zeros((n, d))
    Y = Y.reshape(n, 1)
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d, 1))
    w_old = np.zeros((d, 1))
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0
    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X), 2)
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
        # Update w
        z = w + iters / float(iters + 1) * (w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z) - Y) / L
        for j in range(d):
            w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j]) - lam / L, 0]))

        # Could put in some sort of break condition based on convergence here.
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds], Y)[0]
    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w
