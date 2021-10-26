import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
# import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
from collections import deque
import random
import sys
import os
import math
from math import e

import threading
import time
from time import ctime, sleep

# 信道个数
# N = 3

# 带宽100m
Band = 100

# 簇（或者Agent）的个数
M = 100

# NUMOMA =
NUMMACRO = 5


# RATIO =
# CLUSTER = M - NUMMACRO
CLUSTER = 2

# print("NUMNOMA", NUMNOMA)
# print("NUMMACRO", NUMMACRO)

times = 1

# 定义当前状态的全局变量，方便其他模块实现
# POWER[0]代表簇0的功率
# POWER[1]代表簇1的功率
POWER = [0] * CLUSTER

THETA = [0] * (CLUSTER+NUMMACRO)

POWERMACRO = [0] * NUMMACRO

# RATE[0]代表簇0的速率
# RATE[1]代表簇1的速率
RATE = [0] * CLUSTER

RATEMACRO = [0] * NUMMACRO

# 定义硬件功率
Pc = 5
# 定义一些产生正太分布函数的参数
# mu = 10  # 期望
# sigma = 3  # 标准差为3

rateThreshold = 0

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# 消除GPU的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cost = []

EE = []

REPLACE_TARGET_FREQ = 60  # frequency to update target Q network

class VirtualAgent:
    # 角度
    # theta = 0
    theta_min = 1
    theta_max = 360

    # agent的功率
    P = 0

    P_min = 1
    P_max = 100

    # 离散功率集合的个数
    A = 10
    # 离散角度集合的个数
    B = 5

    miu = 45

    # 从动作集合里选择的动作
    # P_ = 0

    # 角度的增加值
    # theta_ = 0

    # 初始化功率。
    P_list = np.zeros(1 * A, dtype='float64')
    # 指数离散
    # for i in range(A):
    #     P_list[i] = np.array(P_min * (P_max / P_min) ** (i / (A - 1)))
    # 均匀离散
    for i in range(A):
        P_list[i] = np.array((i + 1) * (P_max - P_min) / A)
    print("P_list: ", P_list)

    # 初始化角度。
    # 角度去掉0
    theta_list = np.zeros(1 * B, dtype='float64')
    # 指数式离散
    # for i in range(B):
    #     theta_list[i] = np.array(theta_min * (theta_max / theta_min) ** (i / (B - 1)))
    # 均匀离散
    for i in range(B):
        theta_list[i] = np.array((i + 1) * (theta_max - theta_min) / B)
    print("theta_list:", theta_list)

    # reward计算公式
    # reward =
    # 执行步数。
    step_index = 0

    # 状态数。
    state_num = 2

    # 动作数。假设角度的改变只有两个：不变和增加1度。
    action_num = B * A

    # 簇（或者Agent）的个数
    M = M
    # 带宽
    Band = Band

    # 训练之前观察多少步。
    OBSERVE = 50

    # 选取的小批量训练样本数。
    BATCH = 20

    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    # 以下三个值决定随机选择动作的的概率
    FINAL_EPSILON = 0

    # epsilon 的初始值，epsilon 逐渐减小。
    INITIAL_EPSILON = 0.01

    # epsilon 衰减的总步数。
    EXPLORE = 200

    T = EXPLORE

    # 探索模式计数。
    epsilon = 0.01

    # 训练步数统计。
    learn_step_counter = 0

    # # 学习率。
    learning_rate = 0.1

    # γ经验折损率。
    gamma = 0.9

    # # 迭代次数
    EPOCH = 1

    # 记忆上限。
    memory_size = OBSERVE

    # 当前记忆数。
    memory_counter = 0

    # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    replay_memory_store = deque()

    cumulativeList = []

    agentIndex = []

    cumulativeReward = 0

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

    # np.random.seed(0)  # 随机数种子，相同种子下每次运行生成的随机数相同
    interfereChannelGain = 10 ** (-15) * np.random.uniform(low=0, high=1)
    # H1 = 10 ** (-5) * np.random.uniform(low=0, high=1)  # 基站到IRS信道
    # H2 = 10 ** (-5) * np.random.uniform(low=0, high=1)  # IRS到用户信道

    # print("H1:", H1)
    # print("H2:", H2)
    # print("H:", H)

    def __init__(self, learning_rate=learning_rate, gamma=gamma, memory_size=memory_size):
        self.learning_rate = learning_rate

        self.gamma = gamma
        self.memory_size = memory_size
        # 初始化动作矩阵。
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

        # tf.get_collection(key,scope=None)返回具有给定名称的集合中的值列表
        # 如果未将值添加到该集合，则为空列表。该列表按照收集顺序包含这些值。
        # t_params = tf.get_collection('target_net_params')
        # e_params = tf.get_collection('eval_net_params')
        #
        # # tf.assign(ref,value,validate_shape=None,use_locking=None,name=None)
        # # 该操作在赋值后输出一个张量，该张量保存'ref'的新值。函数完成了将value赋值给ref的作用
        # # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        # 初始化 tensorflow 会话。
        self.session = tf.InteractiveSession()

        # 初始化 tensorflow 参数。
        self.session.run(tf.global_variables_initializer())

        # 记录所有 loss 变化。
        self.cost_his = []

        self.lock = threading.RLock()

    def neuralNetwork(self):
        """
         创建神经网络。
         :return:
        """
        self.current_state_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)  # (?,2)
        self.q_target = tf.placeholder(shape=[None, self.B * self.A], dtype=tf.float32)  # for calculating loss

        with tf.variable_scope('current_net'):
            neural_layer_1 = 8
            w1 = tf.Variable(tf.random_normal([self.state_num, neural_layer_1]))  # (2,8)
            # self.b1 = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)
            b1 = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)
            l1 = tf.sigmoid(tf.matmul(self.current_state_input, w1) + b1)  # q_eval_input (?,2)

            w2 = tf.Variable(tf.random_normal([neural_layer_1, self.action_num]))
            # self.b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            self.q_eval = tf.matmul(l1, w2) + b2

        # neural_layer_2 = 16
        # self.w2 = tf.Variable(tf.random_normal([neural_layer_1, neural_layer_2]))  # (2,8)
        # # self.b1 = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)
        # self.b2 = tf.Variable(tf.zeros([1, neural_layer_2]) + 0.1)
        # l2 = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)  # q_eval_input (?,2)
        #
        # self.w3 = tf.Variable(tf.random_normal([neural_layer_2, self.action_num]))
        # # self.b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        # self.b3 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        # self.q_eval = tf.matmul(l2, self.w3) + self.b3
        self.next_state_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)  # input
        with tf.variable_scope('target_net'):
            neural_layer_1 = 8
            w1t = tf.Variable(tf.random_normal([self.state_num, neural_layer_1]))  # (2,8)
            b1t = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)

            l1t = tf.sigmoid(tf.matmul(self.next_state_input, w1t) + b1t)  # q_eval_input (?,2)

            w2t = tf.Variable(tf.random_normal([neural_layer_1, self.action_num]))
            # self.b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            b2t = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            self.q_next = tf.matmul(l1t, w2t) + b2t

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        # with tf.variable_scope('soft_replacement'):
        # e赋值给t，当前网络的参数赋值给目标网络
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 取出当前动作的得分。
        # self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)

        self.loss = tf.reduce_mean(tf.square((self.q_target - self.q_eval)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def selectAction(self, currentState):
        """
        根据策略选择动作。
        :param currentState: 当前状态
        :return:
        """
        self.lock.acquire()
        # rate, ee = self.getR(self, currentState, agentSeq)
        if np.random.uniform() < self.epsilon:
            # 随机选择
            print("小概率随机选择动作")
            print("currentState: ", currentState)
            current_action_index = np.random.randint(0, self.action_num)
            print("action: ", current_action_index)
        else:
            # 通过神经网络，输出各个actions的value值，选择最大value作为action。
            actions_value = self.session.run(self.q_eval, feed_dict={self.current_state_input: currentState})
            print("currentState: ", currentState)
            # print("actions_value: \n", actions_value)
            action = np.argmax(actions_value)
            print("action: ", action)
            # 得到的只是最大值的index
            current_action_index = action

        # 开始训练后，在 epsilon 小于一定的值之前，将逐步减小 epsilon。
        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
        self.lock.release()
        return current_action_index

    def saveStore(self, currentState, currentActionIndex, ee, currentReward, nextState, cumulativeReward, agentSeq):
        # print("------------进入saveStore！！！------------")
        # 记忆动作(当前状态， 当前执行的动作， 当前动作的得分，下一个状态)。
        self.lock.acquire()
        self.replay_memory_store.append((
            currentState,
            currentActionIndex,
            ee,
            currentReward,
            nextState,
            cumulativeReward,
            agentSeq
        ))

        # print("replay_memory_store  \n     当前状态      |动作索引|奖励|   下个状态 ： ")
        # for i in list(self.replay_memory_store):
        #     print(i, end=" ")
        #     print("\n")

        # 如果超过记忆的容量，则将最久远的记忆移除。
        if len(self.replay_memory_store) > self.memory_size + 1:
            print("弹出旧记忆！")
            self.replay_memory_store.popleft()  # 将头部弹出

        print("当前经验池数目：", len(self.replay_memory_store))
        self.memory_counter += 1
        self.lock.release()

    def getR(self, currentState, agentSeq, isMacro):
        # 不是宏用户，是NOMA簇
        if isMacro == 0:
            # BS天线
            N_BS = 1
            # 接收天线
            N_MS = 1
            #  IRS元件数目
            N_IRS = CLUSTER
            # BS到IRS的路径数目
            L1 = 3
            # IRS到USER的路径数目
            L2 = 3
            # 光速
            c = 3 * 10 ** 8
            # 频率0.34THZ
            f = 0.34 * 10 ** 12
            # 波长
            l = c / f
            # 天线或者elements之间的距离
            d = l / 2

            # np.random.seed(0)
            AOA = np.random.uniform(low=0, high=math.pi)
            AOD = np.random.uniform(low=0, high=math.pi)

            theta = np.array([THETA[agentSeq]] * N_IRS ** 2).reshape(N_IRS, N_IRS)
            # print("theta:", theta)

            A_IRS_1 = (1 / N_IRS ** 0.5) * np.zeros((N_IRS, L1), dtype=np.complex)
            for m in range(L1):  # 列
                for n in range(N_IRS):  # 行
                    A_IRS_1[n][m] = e ** (n * (2j * math.pi / l) * d * math.sin(AOA))

            A_IRS_2 = (1 / N_IRS ** 0.5) * np.zeros((N_IRS, L2), dtype=np.complex)
            for m in range(L2):
                for n in range(N_IRS):
                    A_IRS_2[n][m] = e ** (n * (2j * math.pi / l) * d * math.sin(AOD))

            A_BS = (1 / N_BS ** 0.5) * np.zeros((N_BS, L1), dtype=np.complex)
            for m in range(L1):
                for n in range(N_BS):
                    A_BS[n][m] = e ** (m * (2j * math.pi / l) * d * math.sin(AOD))
            A_MS = (1 / N_BS ** 0.5) * np.zeros((N_MS, L2), dtype=np.complex)
            for m in range(L2):
                for n in range(N_MS):
                    A_MS[n][m] = e ** (m * (2j * math.pi / l) * d * math.sin(AOA))
            # 生成复数对角矩阵
            d = []
            for i in range(L1):
                d.append(np.random.normal(loc=0.0, scale=1) + np.random.normal(loc=0.0, scale=1) * 1j)
            # print(d)
            diag_1 = ((N_BS * N_IRS / L1) ** 0.5) * np.diag(d)
            d.clear()
            # H1 = np.dot(np.dot(A_IRS_1, diag_1), A_BS.conj().T)
            H1 = A_IRS_1 @ diag_1 @ A_BS.conj().T
            # 生成复数对角矩阵
            d = []
            for i in range(L1):
                d.append(np.random.normal(loc=0.0, scale=1) + np.random.normal(loc=0.0, scale=1) * 1j)
            # print(d)
            diag_2 = ((N_MS * N_IRS / L2) ** 0.5) * np.diag(d)
            d.clear()
            # H2 = np.dot(np.dot(A_MS, diag_2), A_IRS_2.conj().T)
            H2 = A_MS @ diag_2 @ A_IRS_2.conj().T
            # H = A_MS @ diag_2 @ A_IRS_2.conj().T @ theta @ A_IRS_1 @ diag_1 @ A_BS.conj().T
            H = H2 @ theta @ H1
            print("H:", H)
            print("abs(H):", abs(H))
            print("POWER:", POWER)

            #  计算SNR
            SNR = (abs(H) * POWER[agentSeq]) / (
                    ((sum(POWER) - POWER[agentSeq]) * self.interfereChannelGain) + (10 ** (-10)) ** 2)

            # print("POWER[0] :", POWER[0] )
            # # print("sum(POWER) - POWER[0] :", sum(POWER) - POWER[0] )
            # print("POWER[agentSeq]:", POWER[agentSeq])
            # print("up:", (self.H2 * self.theta * self.H1) * POWER[agentSeq])
            # print("down:", float(
            #     ((sum(POWER) - POWER[agentSeq]) * self.interfereChannelGain) + 10 ** (-10)))

            print("SNR: ", SNR)
            #  计算速率
            RATE[agentSeq] = Band /(NUMMACRO+CLUSTER)* math.log2(float(1 + SNR))
            print("Rate:", RATE)
            # #  计算reward = 能量效率
            # Psum = Pc + sum(POWER)
            # ee = float(sum(RATE)) / Psum
            # print("ee:", ee)

            # print("self.replay_memory_store[self.agentIndex[-1]][2]:", self.agentIndex)
            # print("self.replay_memory_store[self.agentIndex[-1]][2]:", self.agentIndex[-1][2])

        # 宏用户
        elif isMacro == 1:
            # 宏用户的信道
            h = 10 ** (-5) * np.random.uniform(low=0, high=1)
            SNRMACRO = (h*POWERMACRO[agentSeq-CLUSTER]) / (
                    ((sum(POWERMACRO) - POWERMACRO[agentSeq-CLUSTER]) * self.interfereChannelGain) + (10 ** (-10)) ** 2)
            print("SNRMACRO: ", SNRMACRO)
            RATEMACRO[agentSeq-CLUSTER] = Band /(NUMMACRO+CLUSTER) * math.log2(float(1 + SNRMACRO))
            print("RATEMACRO:", RATEMACRO)

        #  计算能量效率
        print("POWERMACRO:", POWERMACRO)
        Psum = Pc + sum(POWER) + sum(POWERMACRO)
        Rsum = sum(RATE) + sum(RATEMACRO)
        ee = float(Rsum)  - (self.miu * Psum) #eeeee
        # ee = float(Rsum) / Psum
        print("ee:", ee)

        return RATE, RATEMACRO, ee

    def step(self, currentState, actionIndex, agentSeq, isMacro):
        """
        执行动作。
        :param agentSeq:
        :param currentState: 当前状态
        :param actionIndex:  当前动作的Index
        :return:
        """
        self.lock.acquire()
        _, _, ee = self.getR(currentState, agentSeq, isMacro)
        if self.step_index >= 1 and ee > list(self.replay_memory_store)[-1][2]:
            reward = 100 * (ee - self.replay_memory_store[-1][2])  # 当前ee大于上一个ee，正
            # reward = 10
        elif self.step_index >= 1 and ee < list(self.replay_memory_store)[-1][2]:
            reward = 10 * (ee - self.replay_memory_store[-1][2])  # 当前ee小于上一个ee，负
            # reward = -5
        elif self.step_index >= 1 and ee == list(self.replay_memory_store)[-1][2]:
            reward = -1
        else:
            reward = 0
            print("没有ee!")
        print("当前的reward：", reward)
        nextState = np.reshape(self.action_list[actionIndex], (1, 2))
        print("nextState: ", nextState)

        # self.cumulativeReward += (self.gamma ** self.step_index) * reward
        self.cumulativeReward += reward
        self.cumulativeList.append(self.cumulativeReward)
        print("self.cumulativeReward:" + "[" + str(agentSeq) + "] :", self.cumulativeReward)
        self.lock.release()

        return nextState, ee, reward, self.cumulativeReward

    def constrainR(self, currentState, agentSeq, isMacro):
        # 选择动作。
        actionIndex = self.selectAction(currentState)
        # 执行动作，得到：下一个状态，执行动作的得分，是否结束。
        nextState, ee, reward, cumulativeReward = self.step(currentState, actionIndex, agentSeq, isMacro)
        rate, ratemacro, ee = self.getR(nextState, agentSeq, isMacro)

        print("rate:", rate)
        print("ratemacro:", ratemacro)
        if isMacro == 0:
            r = rate[agentSeq]
        elif isMacro ==1 :
            r = ratemacro[agentSeq-CLUSTER]
        while r < rateThreshold:
        # while nextState.all() == currentState.all():
            print("不满足要求，重新选择动作！\n")
            # 选择动作。
            actionIndex = self.selectAction(currentState)
            # 执行动作，得到：下一个状态，执行动作的得分，是否结束。
            nextState, ee, reward, cumulativeReward = self.step(currentState, actionIndex, agentSeq)
            rate, ratemacro, ee = self.getR(nextState, agentSeq, isMacro)
        return actionIndex, nextState, ee, reward, cumulativeReward

    def experience_replay(self):
        """
        记忆回放。
        :return:
        """
        self.lock.acquire()
        print("------------进入经验回放！！！-----------")
        # 随机选择一小批记忆样本。
        # batch = self.BATCH if self.memory_counter > self.BATCH else self.memory_counter
        batch = int(self.OBSERVE / 5)
        print("batch:", batch)
        minibatch = random.sample(self.replay_memory_store, batch)
        # print("minibatch", minibatch)

        batch_state = None
        batch_action = None
        batch_ee = None
        batch_reward = None
        batch_next_state = None
        batch_cumulativeReward = None
        batch_agentSeq = None

        for index in range(len(minibatch)):
            if batch_state is None:
                # 把经验池中的第一个也就是current_state赋给batch_state
                batch_state = minibatch[index][0]
            elif batch_state is not None:
                # 把batch_state, minibatch[index][0])列表合成一个二维数组[ [],[],[] ]
                batch_state = np.vstack((batch_state, minibatch[index][0]))

            if batch_action is None:
                batch_action = minibatch[index][1]
            elif batch_action is not None:
                batch_action = np.vstack((batch_action, minibatch[index][1]))

            if batch_ee is None:
                batch_ee = minibatch[index][2]
            elif batch_ee is not None:
                batch_ee = np.vstack((batch_ee, minibatch[index][2]))

            if batch_reward is None:
                batch_reward = minibatch[index][3]
            elif batch_reward is not None:
                batch_reward = np.vstack((batch_reward, minibatch[index][3]))

            if batch_next_state is None:
                batch_next_state = minibatch[index][4]
            elif batch_next_state is not None:
                batch_next_state = np.vstack((batch_next_state, minibatch[index][4]))

            if batch_cumulativeReward is None:
                batch_cumulativeReward = minibatch[index][5]
            elif batch_cumulativeReward is not None:
                batch_cumulativeReward = np.vstack((batch_cumulativeReward, minibatch[index][5]))

            if batch_agentSeq is None:
                batch_agentSeq = minibatch[index][6]
            elif batch_agentSeq is not None:
                batch_agentSeq = np.vstack((batch_agentSeq, minibatch[index][6]))
            # if batch_done is None:
            #     batch_done = minibatch[index][4]
            # elif batch_done is not None:
            #     batch_done = np.vstack((batch_done, minibatch[index][4]))
        # print("\nbatch_state: \n", batch_state)
        # print("\nbatch_action: \n", batch_action)
        # print("\nbatch_reward: \n", batch_reward)
        # print("\nbatch_next_state: \n", batch_next_state)

        # q_next：下一个状态的 Q 值。
        q_next, q_eval = self.session.run([self.q_next, self.q_eval],
                                          feed_dict={self.current_state_input: batch_state,
                                                     self.next_state_input: batch_next_state})
        # print("q_eval: ", q_eval)
        q_target = q_eval.copy()
        # print("q_target: ", q_target)
        # print("q_target_shape: ", q_target.shape)
        # print("q_next_shape: ", q_next.shape)
        for i in range(len(minibatch)):
            # 当前即时得分。
            current_reward = batch_reward[i][0]
            # current_reward = batch_cumulativeReward[i][0]
            # print(batch_action[i][0])
            q_target[i, batch_action[i][0]] = current_reward + self.gamma * np.max(q_next)
        # print("q_target", q_target)
        # print("q_eval", q_eval)
        # print("q_target-q_eval:", q_target - q_eval)
        #     print(np.max(q_next))
        # print("q_target_shape: ", q_target.shape)
        # print("batch_state: ", batch_state.shape)
        # q_target_new = np.expand_dims(q_target, axis=1)
        # # print("q_target_new: ", q_target_new)
        #
        # # 变为和q_eval相同形状
        # q_target_new1 = np.zeros((batch, self.B * self.A), dtype='float32')
        # for i in range(batch):  # 行
        #     for j in range(2 * self.A):  # 列
        #         q_target_new1[i][j] = q_target_new[i][0]

        # print("q_target_new1: \n", q_target_new1)

        # print("shapes! ! ! ")
        # print(self.q_eval_input.shape)  # (?, 2)
        # print(batch_state.shape)  # (数据条数, 2)

        # print(self.action_input.shape)  # (?,action_num) = (?, 10)
        # print(batch_action.shape)  # (2, 1)
        # print("self.action_list[batch_action]", self.action_list[batch_action])
        # print(self.action_list[batch_action].shape)  # (2, 1, 2)

        # print(self.q_target.shape)  # (?,)
        # print(q_target.shape)

        # output = self.session.run(self.q_eval, feed_dict={self.q_eval_input: batch_state})
        # print("output: ", output)

        # print("batch_state: ", batch_state)

        _, cost = self.session.run([self.train_op, self.loss],
                                   feed_dict={self.current_state_input: batch_state,
                                              self.q_target: q_target})

        # print("q_e:", q_e)
        # print("learning_rate:", self.session.run(self.learning_rate))
        # learning_rate1 = self.session.run(self.learning_rate)
        # self.learning_rate_list.append(learning_rate1)
        # print("w1:", self.session.run(self.w1))
        # print("b1:", self.session.run(self.b1))
        # print("w2:", self.session.run(self.w2))
        # print("b2:", self.session.run(self.b2))
        self.cost_his.append(cost)
        self.lock.release()

    def deleteCurrentStateFromActionlist(self,currentState):
        # print(np.where(self.action_list == currentState))
        lst = list(np.where(self.action_list == currentState)[0])
        # print(type(lst))
        row = 0
        for i in range(len(lst)):
            num = lst.count(lst[i])
            if num == 2:
                row = lst[i]
                print(lst[i])
                break
        self.action_list = np.delete(self.action_list,row,axis=0)
        print("已经从动作集合中删除当前状态！")
        # print(self.action_list)

    def train(self, agentSeq, isMacro):
        self.lock.acquire()
        for episode in range(self.EPOCH):
            print("self.action_list: \n", self.action_list)
            # 初始化当前状态。
            currentState = self.state_list
            print("currentState", currentState)
            # self.epsilon = self.INITIAL_EPSILON print("\n第__" + str(agentSeq) + "__agent" + " start time: ",
            # ctime()) print("\n第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练 Initial POWER:" +
            # str(POWER) + "\n")
            reward_counter = 0
            # self.agentIndex = []
            THETA[agentSeq] = currentState[0][0]

            if isMacro == 0:
                POWER[agentSeq] = currentState[0][1]
            elif isMacro == 1:
                POWERMACRO[agentSeq-CLUSTER] = currentState[0][1]

            self.step_index = 0
            for t in range(self.T):
                # self.deleteCurrentStateFromActionlist(currentState)
                print("\n第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练：")
                actionIndex, nextState, ee, reward, cumulativeReward = self.constrainR(currentState, agentSeq, isMacro)
                # 保存记忆。
                self.saveStore(currentState, actionIndex, ee, reward, nextState, cumulativeReward, agentSeq)

                # print("\nself.replay_memory_store[self.step_index][2]:", self.replay_memory_store[self.step_index][2])
                # print("\nself.replay_memory_store[self.step_index-1][2]:", self.replay_memory_store[self.step_index-1][2])

                # 每self.OBSERVE次，经验回放一次

                if self.step_index > 0 and self.step_index >= self.OBSERVE :
                    # if self.step_index % self.OBSERVE == 0:
                    # self.printMemory()
                    self.experience_replay()
                    # self.OBSERVE += self.step_index - self.OBSERVE
                    # print("self.OBSERVE:", self.OBSERVE)

                #  每REPLACE_TARGET_FREQ次后，更新target网络
                if self.step_index % REPLACE_TARGET_FREQ == 0:
                    print("更新一次target网络！！！")
                    self.session.run(self.target_replace_op)

                # # 在经验池中找到当前agent的数据的索引，即第几条数据是这个agent的
                # for i, item in enumerate(list(self.replay_memory_store)):
                #     if item[-1] == agentSeq:
                #         self.agentIndex.append(i)
                #     #  由于只需要最后一个值，前面的值可以删掉，防止列表溢出
                #     if self.step_index > 10 and len(self.agentIndex) > 10:
                #         self.agentIndex.pop(0)
                # print("self.agentIndex:", self.agentIndex)

                #  如果reward 一直为0， reward_counter累加到一定值
                # if reward == -1:
                #     reward_counter += 1
                # else:
                #     reward_counter = 0

                # 更新状态
                currentState = nextState
                # if t == self.T-1 :
                #     global EE
                #     EE.append(ee)

                #  如果循环了100次，将增大随机选择的概率
                # if reward_counter == 100:
                #     print("增大随机选择的概率!")
                #     self.INITIAL_EPSILON = 0.8

                # 训练停止条件
                # if 0 < list(self.replay_memory_store)[agentIndex[-1]][5] \
                #         - list(self.replay_memory_store)[agentIndex[-2]][5] < threshold:

                # if len(self.agentIndex) > 0 and len(self.replay_memory_store) > 0:
                #     #  总的奖励大于10
                #     #  某一次的即时奖励大于10
                #     #  某个状态循环了100次
                #     print("判断停止条件！")
                #     print("reward_counter: ", reward_counter)
                #     # if list(self.replay_memory_store)[self.agentIndex[-1]][2] >= 30\
                #     #         or list(self.replay_memory_store)[self.agentIndex[-1]][5] >= 30 \
                #     #         or self.step_index >= 10000:

                # print("------------训练停止！！！！！！-----------")
                # print("reward_counter: ", reward_counter)
                # # print("replay_memory_store  \n     当前状态      |动作索引|奖励|   下个状态 ： ")
                # # for i in list(self.replay_memory_store):  # 转成list，否则deque迭代会产生错误
                # #     print(i, end=" ")
                # #     print("\n")
                # # print(str(list(self.replay_memory_store)[agentIndex[-1]][4]) +
                # #       "-" + str(list(self.replay_memory_store)[agentIndex[-2]][4]) + "<" + str(threshold))
                # # print("agentSeq:", agentSeq)
                # # print("agentIndex:", agentIndex)
                # # 转成list，否则deque迭代会出错
                # # 当cumulativeReward增加的值小于阈值，停止训练
                # EE.append((sum(RATE) / (sum(POWER) + Pc)))
                # print("第__" + str(agentSeq) + "__agent 训练完成\n" + "  POWER:"
                #       + str(POWER[agentSeq]) + "," + "  角度:" + str(currentState[0][0])
                #       + "," + "  EE:" + str(sum(RATE) / (sum(POWER) + Pc)) + "  POWER:" + str(
                #     POWER) + "  RATE:" + str(RATE) + "\n")
                # # print("第__" + str(agentSeq) + "__agent 训练完成的 cost_his: ", self.cost_his)
                # global cost
                # cost.append(self.cost_his)
                # # self.plotCumulativeReward()
                # print("cost:", cost)
                # #  数据清零
                # self.step_index = 0
                # # self.cumulativeReward = 0
                # # self.cost_his.clear()
                # # self.agentIndex.clear()
                # self.cumulativeReward = 0
                # # time.sleep(8)
                # # self.replay_memory_store.clear()
                # print("\n")
                # break  # 跳出while循环

                # print("第" + str(agentSeq) + " agent "+"第" + str(self.step_index) + "次训练POWER:", POWER)
                if isMacro == 0:
                    print("第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练完角度:", currentState[0][0])
                    print("第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练完POWER:", currentState[0][1])
                else:
                    print("第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练完角度:", currentState[0][0])
                    print("第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练完POWER:", currentState[0][1])
                self.step_index += 1
                print("reward_counter:", reward_counter)

            EE.append(ee)
            self.getEE()

            #  end for
            # print("第__" + str(agentSeq) + "__agent 训练完成的 cost_his: ", self.cost_his)

            # plt.plot(np.arange(len(self.cost_his)), self.cost_his, 'b-')
            # plt.ylabel('Cost')
            # plt.xlabel('training steps')
            # plt.show()

        # end for
        self.lock.release()
    def getEE(self):
        #  训练的结果
        EE_ = []
        global EE
        #  去掉相同元素
        # EE = list(set(EE))
        print("EE:", EE)
        #  最后训练完的才是系统的EE，取最后一个
        for item in range(len(EE)):
            if item % (NUMMACRO + CLUSTER) == NUMMACRO + CLUSTER - 1 :
                EE_.append(EE[item])
        print("EE_:", EE_)
        # EE 是每个Agent训练结束时的能量效率，由于有延迟，以最后一个训练完的EE为准，即EE_
        # 将EE_取平均
        # averageEE = []
        # # 每avergeTime个元素之和
        # s = 0
        # avergeTime = CLUSTER + NUMMACRO - 1
        # for index in range(1, len(EE_) + 1):
        #     if index % avergeTime == 0:
        #         for m in range(index - avergeTime, index):
        #             s += EE_[m]
        #         averageEE.append(float(s / avergeTime))
        # averageEE.append(sum(EE_) / len(EE_))
        # print("训练平均值:", averageEE)

    def compareWithRandom (self):
        #  随机的结果
        randomPower = []
        randomTheta = []
        macrorandomPower = []
        macrorandomTheta = []

        # random.seed(0)
        for i in range(CLUSTER):
            randomPower.append(np.random.uniform(self.P_min, self.P_max))
            randomTheta.append(np.random.uniform(self.theta_min, self.theta_max))

        for i in range(NUMMACRO):
            macrorandomPower.append(np.random.uniform(self.P_min, self.P_max))
            macrorandomTheta.append(np.random.uniform(self.theta_min, self.theta_max))

        RATE = [0] * CLUSTER
        rateMACRO = [0] * NUMMACRO

        ee = []

        # BS天线
        N_BS = 1
        # 接收天线
        N_MS = 1
        #  IRS元件数目
        N_IRS = CLUSTER
        # BS到IRS的路径数目
        L1 = 3
        # IRS到USER的路径数目
        L2 = 3
        # 光速
        c = 3 * 10 ** 8
        # 频率0.34THZ
        f = 0.34 * 10 ** 12
        # 波长
        l = c / f
        # 天线或者elements之间的距离
        d = l / 2
        # np.random.seed(0)
        AOA = np.random.uniform(low=0, high=math.pi)
        AOD = np.random.uniform(low=0, high=math.pi)
        theta = np.array([np.random.uniform(low=self.theta_min, high= self.theta_max)] * N_IRS ** 2).reshape(N_IRS, N_IRS)
        print("theta:",theta)
        # print("theta:", theta)
        A_IRS_1 = (1 / N_IRS ** 0.5) * np.zeros((N_IRS, L1), dtype=np.complex)
        for m in range(L1):  # 列
            for n in range(N_IRS):  # 行
                A_IRS_1[n][m] = e ** (n * (2j * math.pi / l) * d * math.sin(AOA))
        A_IRS_2 = (1 / N_IRS ** 0.5) * np.zeros((N_IRS, L2), dtype=np.complex)
        for m in range(L2):
            for n in range(N_IRS):
                A_IRS_2[n][m] = e ** (n * (2j * math.pi / l) * d * math.sin(AOD))
        A_BS = (1 / N_BS ** 0.5) * np.zeros((N_BS, L1), dtype=np.complex)
        for m in range(L1):
            for n in range(N_BS):
                A_BS[n][m] = e ** (m * (2j * math.pi / l) * d * math.sin(AOD))
        A_MS = (1 / N_BS ** 0.5) * np.zeros((N_MS, L2), dtype=np.complex)
        for m in range(L2):
            for n in range(N_BS):
                A_MS[n][m] = e ** (m * (2j * math.pi / l) * d * math.sin(AOA))
        # 生成复数对角矩阵
        d = []
        for i in range(L1):
            d.append(np.random.normal(loc=0.0, scale=1) + np.random.normal(loc=0.0, scale=1) * 1j)
        # print(d)
        diag_1 = ((N_BS * N_IRS / L1) ** 0.5) * np.diag(d)
        d.clear()
        # H1 = np.dot(np.dot(A_IRS_1, diag_1), A_BS.conj().T)
        H1 = A_IRS_1 @ diag_1 @ A_BS.conj().T
        # 生成复数对角矩阵
        d = []
        for i in range(L1):
            d.append(np.random.normal(loc=0.0, scale=1) + np.random.normal(loc=0.0, scale=1) * 1j)
        # print(d)
        diag_2 = ((N_MS * N_IRS / L2) ** 0.5) * np.diag(d)
        d.clear()
        # H2 = np.dot(np.dot(A_MS, diag_2), A_IRS_2.conj().T)
        H2 = A_MS @ diag_2 @ A_IRS_2.conj().T
        # H = A_MS @ diag_2 @ A_IRS_2.conj().T @ theta @ A_IRS_1 @ diag_1 @ A_BS.conj().T
        H = H2 @ theta @ H1
        print("H:", H)
        print("abs(H):", abs(H))


        print("开始计算NOMA用户")
        for i in range(CLUSTER): # 前CLUSTER个为NOMA用户
            print("POWER:", POWER)
            #  计算SNR
            SNR = (abs(H) * randomPower[i]) / (
                    ((sum(randomPower) - randomPower[i]) * self.interfereChannelGain) + (10 ** (-10)) ** 2)
            # print("POWER[0] :", POWER[0] )
            # # print("sum(POWER) - POWER[0] :", sum(POWER) - POWER[0] )
            # print("POWER[agentSeq]:", POWER[agentSeq])
            # print("up:", (self.H2 * self.theta * self.H1) * POWER[agentSeq])
            # print("down:", float(
            #     ((sum(POWER) - POWER[agentSeq]) * self.interfereChannelGain) + 10 ** (-10)))
            print("SNR: ", SNR)
            #  计算速率
            RATE[i] = Band / (NUMMACRO + CLUSTER) * math.log2(float(1 + SNR))
            print("Rate:", RATE)
            # #  计算reward = 能量效率
            # Psum = Pc + sum(POWER)
            # ee = float(sum(RATE)) / Psum
            # print("ee:", ee)
            # print("self.replay_memory_store[self.agentIndex[-1]][2]:", self.agentIndex)
            # print("self.replay_memory_store[self.agentIndex[-1]][2]:", self.agentIndex[-1][2])


        # 宏用户
        print("开始计算宏用户")
        for i in range(CLUSTER, M):  # 前CLUSTER个为宏用户
            # 宏用户的信道i
            h = 10 ** (-2) * np.random.uniform(low=0, high=1)
            SNRMACRO = (h * macrorandomPower[i - CLUSTER]) / (
                    ((sum(macrorandomPower) - macrorandomPower[i - CLUSTER]) * self.interfereChannelGain) + (
                        10 ** (-10)) ** 2)
            print("SNRMACRO: ", SNRMACRO)
            rateMACRO[i - CLUSTER] = Band / (NUMMACRO + CLUSTER) * math.log2(float(1 + SNRMACRO))
            print("RATEMACRO:", rateMACRO)
        #  计算能量效率
        print("POWERMACRO:", macrorandomPower)
        Psum = Pc + sum(randomPower) + sum(macrorandomPower)
        Rsum = sum(RATE) + sum(rateMACRO)
        ee.append(float(Rsum) / Psum)
        print("ee:", ee)

    def plotCumulativeReward(self):
        #  画出CumulativeReward
        print("cumulativeList:", self.cumulativeList)
        plt.plot(np.arange(len(self.cumulativeList)), self.cumulativeList, 'g-')
        plt.ylabel('CumulativeReward')
        plt.xlabel('训练次数')
        plt.show()
        self.cumulativeList.clear()

    def printMemory(self):
        print("replay_memory_store  \n     当前状态               |动作索引|     ee       |奖励|    下个状态 ： ")
        for i in list(self.replay_memory_store):  # 转成list，否则deque迭代会产生错误
            print(i, end=" ")
            print("\n")


def plot_cost():
    for i in range(M):
        plt.figure(i)
        plt.plot(np.arange(len(cost)), cost, 'b-')
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def plot_EE():
    global EE8
    # EE = list(set(EE))
    print("EE:", EE)
    plt.plot(np.arange(len(EE)), EE, 'r-')
    plt.ylabel('次数')
    plt.xlabel('EE')
    plt.show()


if __name__ == '__main__':
    #  标志位
    threadOn = 1
    compareOn = 1
    #  多次训练画出EE
    EEWithTimes = 1
    # #
    # # if threadOn == 0 and compareOn == 0 and EEWithTimes == 1:
    # #     for i in range(times):  # 多次训练
    # #         print("第" + str(i) + "次训练！")
    # #         #  实例化
    # #         #  M为Agent个数
    # #         virtualAgent = []
    # #         for i in range(M):
    # #             virtualAgent.append(VirtualAgent())
    # #             # for j in range(100):
    # #             # virtualAgent[i].compareWithRandom()
    # #             # virtualAgent[i].train(i)
    # #
    # #         threads = []
    # #         t = []
    # #         for i in range(M):
    # #             t.append(threading.Thread(target=virtualAgent[i].train, args=(i,)))
    # #             # t0 = threading.Thread(target=virtualAgent_0.train, args=(0,))
    # #             # t1 = threading.Thread(target=virtualAgent_1.train, args=(1,))
    # #             # t2 = threading.Thread(target=virtualAgent_2.train, args=(2,))
    # #
    # #         # t1 = threading.Thread(target=virtualAgent_0.train(0))
    # #         # t1 = threading.Thread(target=virtualAgent_0.train(), args=(0,))
    # #         # 这样写无法实现多线程
    # #
    # #         for i in range(M):
    # #             threads.append(t[i])
    # #
    # #         for item in threads:
    # #             item.setDaemon(True)
    # #             item.start()
    # #
    # #         for item in threads:
    # #             item.join()
    # #
    # #         print("EE:", EE)
    # #         print("第" + str(i) + "次训练结束！")
    # #
    # #     #  子线程执行完毕，在主线程里画图
    # #     plot_EE()
    # #     EE.clear()
    # #
    # # if threadOn == 1 and compareOn == 0 and EEWithTimes == 0:
    # #     # for tim in range(times):
    # #     print("执行多线程！")
    # #     #  实例化
    # #     #  M为Agent个数
    # #     virtualAgent = []
    # #     threads = []
    # #     t = []
    # #
    # #     for i in range(M):
    # #         virtualAgent.append(VirtualAgent())
    # #         t.append(threading.Thread(target=virtualAgent[i].train, args=(i,)))
    # #
    # #         # for j in range(100):
    # #         # virtualAgent[i].compareWithRandom()
    # #         # virtualAgent[i].train(i)
    # #     # t0 = threading.Thread(target=virtualAgent_0.train, args=(0,))
    # #     # t1 = threading.Thread(target=virtualAgent_1.train, args=(1,))
    # #     # t2 = threading.Thread(target=virtualAgent_2.train, args=(2,))
    # #
    # #     # t1 = threading.Thread(target=virtualAgent_0.train(0))
    # #     # t1 = threading.Thread(target=virtualAgent_0.train(), args=(0,))
    # #     # 这样写无法实现多线程
    # #
    # #     for i in range(M):
    # #         threads.append(t[i])
    # #
    # #     for item in threads:
    # #         item.setDaemon(True)
    # #         item.start()
    # #
    # #     for item in threads:
    # #         item.join()
    # #
    # #     # virtualAgent[0].plotCumulativeReward()
    # #     # print("主线程cost:", cost)
    # #     #  子线程执行完毕，在主线程里画图
    # #     # plot_cost()
    # #     cost.clear()
    # #
    # # if threadOn == 0 and compareOn == 0 and EEWithTimes == 0:
    # #     print("不执行线程！")
    # #     #  实例化
    # #     #  M为Agent个数
    # #     for tim in range(times):
    # #         virtualAgent = []
    # #         for i in range(M):
    # #             virtualAgent.append(VirtualAgent())
    # #             if i < NUMNOMA:
    # #                 virtualAgent[i].train(agentSeq=i, isMacro=0)
    # #             else:
    # #                 virtualAgent[i].train(agentSeq=i, isMacro=1)
    # #             # virtualAgent[i].plotCumulativeReward()
    # #         # plot_cost()
    # #         print("第" + str(tim) + "次训练结束！")
    # #
    # # if threadOn == 0 and compareOn == 1 and EEWithTimes == 0:
    # #     print("不执行线程，之间比较！")
    # #     #  实例化
    # #     #  M为Agent个数
    # #     virtualAgent = []
    # #     for i in range(M):
    # #         virtualAgent.append(VirtualAgent())
    # #     virtualAgent[0].compareWithRandom()
    # #
    if threadOn == 1 and compareOn == 1 and EEWithTimes == 1:

        for tim in range(times):  # 多次训练
            print("开始DQN训练")
            print("第" + str(tim) + "次大的训练！")

            #  实例化
            #  M为Agent个数
            virtualAgent = []
            threads = []
            t = []

            # 宏用户/ NOAM簇 =  5
            # M = 宏用户 + NOAM簇，M为6的倍数
            # 一共 NUMmacro + ___ 个线程
            for i in range( NUMMACRO+CLUSTER ):
                virtualAgent.append(VirtualAgent())
                # for j in range(100):
                # virtualAgent[i].compareWithRandom()
                # virtualAgent[i].train(i)
                # 前NUMNOMA是noma用户，但只使用一个线程
                if i < CLUSTER :
                    isMacro = 0
                # 后NUMMACRO是宏用户，使用NUMMACRO个线程
                elif i >= CLUSTER:
                    # 宏用户:
                    isMacro = 1
                print("i:",i)
                t.append(threading.Thread(target=virtualAgent[i].train, args=(i, isMacro)))
                # t0 = threading.Thread(target=virtualAgent_0.train, args=(0,))
                # t1 = threading.Thread(target=virtualAgent_1.train, args=(1,))
                # t2 = threading.Thread(target=virtualAgent_2.train, args=(2,))

                # t1 = threading.Thread(target=virtualAgent_0.train(0))
                # t1 = threading.Thread(target=virtualAgent_0.train(), args=(0,))
                # 这样写无法实现多线程
                threads.append(t[i])

            for item in threads:
                item.setDaemon(True)
                item.start()

            for item in threads:
                item.join()

            # print("EE:", EE)
            print("第" + str(tim) + "次训练结束！")

            print("结束第" + str(tim) + "次大的训练！")
        print("训练结束！")

        # print("开始随机计算")
        # #  实例化
        # #  M为Agent个数
        # virtualAgent = []
        # for i in range(M):
        #     virtualAgent.append(VirtualAgent())
        # virtualAgent[0].compareWithRandom()

    # agent = VirtualAgent()
    # agent.compareWithRandom()


