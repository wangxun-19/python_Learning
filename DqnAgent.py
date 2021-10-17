import matplotlib as plt
import tensorflow.compat.v1 as tf
import numpy as np
from collections import deque
import random
import sys
import os
import math

import threading

# 信道个数
N = 3
# 簇（或者Agent）的个数
M = 1
# 带宽20m
Band = 20

POWER = [0] * M

RATE = [0] * M

Pc = 5
threshold = 10 ** (-4)

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# 消除GPU的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cost = []

EE = []

times = 100  # 训练次数画图时的横坐标

REPLACE_TARGET_FREQ = 200  # frequency to update target Q network

class DeepWork :
    #角度
    theta = 0
    theta_min = 1
    theta_max = 360
    #功率
    P = 0
    P_min =1
    P_max = 100
    # 离散功率集合的个数
    A = 10
    # 离散角度集合的个数
    B = 5

    P_list = np.zeros(1*A, dtype='float64')

    for i in range(A):
        P_list[i] = np.array((i + 1) * (P_max - P_min) / A)
    print("P_list: ", P_list)

    theta_list = np.zeros(1 * B, dtype='float64')
    for i in range(B):
        theta_list[i] = np.array((i + 1) * (theta_max - theta_min) / B)
    print("theta_list:", theta_list)

    #执行步数
    step_index = 0
    #状态数
    state_num = 2

    #R - μP
    miu = 0.5

    # 动作数。假设角度的改变只有两个：不变和增加1度。
    action_num = B * A

    N = N
    # 簇（或者Agent）的个数
    M = M
    # 带宽
    Band = Band

    # 训练之前观察多少步。
    OBSERVE = 100

    # 选取的小批量训练样本数。
    BATCH = 20

    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    # 以下三个值决定随机选择动作的的概率
    FINAL_EPSILON = 0

    # epsilon 的初始值，epsilon 逐渐减小。
    INITIAL_EPSILON = 0.01

    # epsilon 衰减的总步数。
    EXPLORE = 3000.

    # 探索模式计数。
    epsilon = 0.

    # 训练步数统计。
    learn_step_counter = 0

    # # 学习率。
    learning_rate = 0.9

    # γ经验折损率。
    gamma = 0.999

    # # 迭代次数
    # EPOCH = 500

    # 记忆上限。
    memory_size = 100000

    # 当前记忆数。
    memory_counter = 0

    # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    replay_memory_store = deque()

    cumulativeList = []

    agentIndex = []

    # 生成一个状态矩阵。
    state_list = None

    # 生成一个动作矩阵。
    action_list = None

    # q_eval 网络。
    # 网络的输入
    q_eval_input = None
    # 执行的动作
    action_input = None
    # 网络的输出（执行动作的Q值）
    q_eval = None

    q_target = None
    predict = None

    loss = None
    train_op = None
    cost_his = None
    reward_action = None

    # tensorflow 会话。
    session = None

    np.random.seed(0)  # 随机数种子，相同种子下每次运行生成的随机数相同
    interfereChannelGain = 10 ** (-15) * np.random.uniform(low=0, high=1)
    H1 = 10 ** (-5) * np.random.uniform(low=0, high=1)  # 基站到IRS信道
    H2 = 10 ** (-5) * np.random.uniform(low=0, high=1)  # IRS到用户信道

    def __init__(self, learning_rate=learning_rate, gamma=gamma, memory_size=memory_size):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.action_list = np.zeros((self.B * self.A, 2), dtype='float64')
        k = 0
        for i in range(len(self.theta_list)):
            for j in range(len(self.P_list)):
                self.action_list[k] = np.array([self.theta_list[i], self.P_list[j]])
                if k < (self.B * self.A):
                    k = k + 1

                # 初始化状态矩阵。
            self.state_list = np.zeros((1, 2), dtype='float64')

            #  随机初始化状态
            self.state_list[0] = self.action_list[random.randint(0, len(self.action_list) - 1)]

            # 创建神经网络。
            self.neuralNetwork()

            # 初始化 tensorflow 会话。
            self.session = tf.InteractiveSession()

            # 初始化 tensorflow 参数。
            self.session.run(tf.global_variables_initializer())

            # 记录所有 loss 变化。
            self.cost_his = []

            self.cumulativeReward = 0

            self.lock = threading.RLock()

if __name__ == "__main__" :
    q_network = '123'
    print(q_network)