from pde import *
import warnings
import sys
import datetime
from setup import use_difference
warnings.filterwarnings('ignore')

# SGA-PDE的主要程序, 定义 SGA 类
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class SGA:  # 最外层
    def __init__(self, num, depth, width, p_var, p_mute, p_rep, p_cro):
        """
            1. num: pool里PDE的数量
            2. depth: 每个PDE的term的最大深度
            3. width: 每个PDE所含term的最大数量
            4. p_var: 生成树时节点为u/t/x而不是运算符的概率
            5. p_rep: 将（所有）pde某一项重新生成以替换原项的概率
            6. p_mute: PDE的树结构里每个节点的突变概率
            7. p_cro: 不同PDE之间交换term的概率
        """
        self.num = num
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.p_rep = p_rep
        self.eqs = []
        self.mses = []
        self.ratio = 1
        self.repeat_cross = 0
        self.repeat_change = 0
        print('Creating the original pdes in the pool ...')
        # 循环产生num个pde
        for i in range(num * self.ratio):
            a_pde = PDE(depth, width, p_var)
            a_err, a_w = evaluate_mse(a_pde)
            pde_lib.append(a_pde)
            err_lib.append((a_err, a_w))
            while a_err < -100 or a_err == np.inf:  # MSE太小则直接去除，to avoid u d t
                print('a_err: {} 太小!'.format(a_err))
                a_pde = PDE(depth, width, p_var)
                a_err, a_w = evaluate_mse(a_pde)
                pde_lib.append(a_pde)
                err_lib.append((a_err, a_w))
            print('Creating the ith pde, i=', i)
            print('a_pde.visualize():', a_pde.visualize())
            print('evaluate_aic:', a_err)
            self.eqs.append(a_pde)
            self.mses.append(a_err)

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse) # 从小到大排序，提取出排序的index
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        self.mses, self.eqs = self.mses[0:num], self.eqs[0:num]

    def run(self, gen=100):
        for i in range(1, gen+1):
            self.cross_over(self.p_cro)
            self.change(self.p_mute, self.p_rep)
            best_eq, best_mse = self.the_best()
            print('{} generation best_aic & best Eq: {}, {}'.format(i, best_mse, best_eq.visualize()))
            print('best concise Eq: {}'.format(best_eq.concise_visualize()))
            if best_mse < 0:
                print('We are close to the answer, pay attention')
            print('{} generation repeat cross over {} times and mutation {} times'.
                  format(i, self.repeat_cross, self.repeat_change))
            self.repeat_cross, self.repeat_change = 0, 0

    def the_best(self):
        argmin = np.argmin(self.mses)
        return self.eqs[argmin], self.mses[argmin]

    # 交叉操作: 比如一代有2n个样本，先用最好的n个样本交叉，产生了m个新的不重复的样本。则最终提取了2n+m个样本中最好的2n个
    def cross_over(self, percentage=0.5):
        def cross_individual(pde1, pde2):
            new_pde1, new_pde2 = copy.deepcopy(pde1), copy.deepcopy(pde2)
            w1, w2 = len(pde1.elements), len(pde2.elements)
            ix1, ix2 = np.random.randint(w1), np.random.randint(w2)
            new_pde1.elements[ix1] = pde2.elements[ix2]
            new_pde2.elements[ix2] = pde1.elements[ix1]
            return new_pde1, new_pde2

        # 一半的好样本保存，并在此基础上交叉生成一半新样本
        # print('begin crossover')
        num_ix = int(self.num * percentage)
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        copy_mses, copy_eqs = self.mses[0:num_ix], self.eqs[0:num_ix]  # top percentage samples

        new_eqs, new_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        reo_eqs, reo_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        random.shuffle(reo_mse)
        random.shuffle(reo_eqs)

        for a, b in zip(new_eqs, reo_eqs):
            new_a, new_b = cross_individual(a, b) # 在好样本的基础上交叉
            if new_a.visualize() in pde_lib:
                self.repeat_cross += 1
            else: # 前一半样本交叉产生了新的pde，则加入lib中，并且加入当前代的全部样本中
                a_err, a_w = evaluate_mse(new_a)
                pde_lib.append(new_a.visualize())
                err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_a)

            if new_b.visualize() in pde_lib:
                self.repeat_cross += 1
            else: # 前一半样本交叉产生了新的pde，则加入lib中，并且加入当前代的全部样本中
                b_err, b_w = evaluate_mse(new_b)
                pde_lib.append(new_b.visualize())
                err_lib.append((b_err, b_w))
                self.mses.append(b_err)
                self.eqs.append(new_b)

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]

    def change(self, p_mute=0.05, p_rep=0.3):
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)

        for i in range(self.num):
            # 保留最好的那部分eqs不change，只cross over.
            if i < 1: #保留最好的1个样本，不进行change
                continue
            # print(self.eqs[i].visualize())
            new_eqs[i].mutate(p_mute)
            replace_or_not = np.random.choice([False, True], p=([1 - p_rep, p_rep]))
            if replace_or_not:
                new_eqs[i].replace()
            # 检测是否重复
            if new_eqs[i].visualize() in pde_lib:
                self.repeat_change += 1
            else:
                a_err, a_w = evaluate_mse(new_eqs[i])
                pde_lib.append(new_eqs[i].visualize())
                err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_eqs[i])

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]


if __name__ == '__main__':
    seed = config.seed
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sga_num = 20
    sga_depth = 4
    sga_width = 5
    sga_p_var = 0.5
    sga_p_rep = 1
    sga_p_mute = 0.3
    sga_p_cro = 0.5
    sga_run = 100

    log_filename = f'output/sga/{config.problem}/pop{sga_num}_depth{sga_depth}_width{sga_width}_var{sga_p_var}_replace{sga_p_rep}_mute{sga_p_mute}_cross{sga_p_cro}_run{sga_run}_seed{seed}-diff{use_difference}.log'
    sys.stdout = Logger(log_filename, sys.stdout)

    print('current_time: ', current_time)
    print('problem: ', config.problem)
    print('use_difference: ', use_difference)
    print('sga_num = ', sga_num)
    print('sga_depth = ', sga_depth)
    print('sga_width = ', sga_width)
    print('sga_p_var = ', sga_p_var)
    print('sga_p_rep = ', sga_p_rep)
    print('sga_p_mute = ', sga_p_mute)
    print('sga_p_cro = ', sga_p_cro)
    print('sga_run = ', sga_run)

    sga = SGA(num=sga_num, depth=sga_depth, width=sga_width, p_var=sga_p_var, p_rep=sga_p_rep, p_mute=sga_p_mute, p_cro=sga_p_cro)
    sga.run(sga_run)

# 超参数设定
# num = 20 # the population of each generation is 20.

# depth = 4 # the maximum tree depth hyperparameter to constrain the number of nesting calculations in each function term.

# width = 5 # the maximum forest width hyperparameter to constrain the number of function terms in the PDE.

# p_var = 0.5 # The probability of generating a node as an operand (leaf node) instead of an operator is 0.5.

# p_mute = 0.3 # The probability of mutation at each node is 0.3.

# p_cro = 0.5 # The probability of crossover between different function terms in two PDE is 0.5.

# p_rep = 1 # The probability of regenerating a term to replace the original term in the PDE.

# sga_run = 100 # the maximum generation is set to 100.
