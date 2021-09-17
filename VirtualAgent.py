import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
from collections import deque
import random
import sys
import os
import math

import threading
import time
from time import ctime, sleep

# 信道个数
N = 3
# 簇（或者Agent）的个数
M = 1
# 带宽20m
Band = 20

# 定义当前状态的全局变量，方便其他模块实现
# POWER[0]代表agent0的功率
# POWER[1]代表agent1的功率
POWER = [0] * M

# RATE[0]代表agent0的速率
# RATE[1]代表agent1的速率
RATE = [0] * M
# 定义硬件功率
Pc = 5
# 定义一些产生正太分布函数的参数
# mu = 10  # 期望
# sigma = 3  # 标准差为3

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


class VirtualAgent:
    # 角度
    theta = 0
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

    # 信道个数
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
    epsilon = 0

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

        # 初始化 tensorflow 会话。
        self.session = tf.InteractiveSession()

        # 初始化 tensorflow 参数。
        self.session.run(tf.global_variables_initializer())

        # 记录所有 loss 变化。
        self.cost_his = []

        self.cumulativeReward = 0

        self.lock = threading.RLock()

    def neuralNetwork(self):
        """
         创建神经网络。
         :return:
        """
        self.state_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)  # (?,2)
        with tf.variable_scope('current_net'):
            neural_layer_1 = 8
            w1 = tf.Variable(tf.random_normal([self.state_num, neural_layer_1]))  # (2,8)
            # self.b1 = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)
            b1 = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)

            l1 = tf.sigmoid(tf.matmul(self.state_input, w1) + b1)  # q_eval_input (?,2)

            w2 = tf.Variable(tf.random_normal([neural_layer_1, self.action_num]))
            # self.b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            self.q_eval = tf.matmul(l1, w2) + b2


        with tf.variable_scope('target_net'):
            # self.q_target = tf.placeholder(shape=[None, self.B * self.A], dtype=tf.float32)

            neural_layer_1 = 8
            w1t = tf.Variable(tf.random_normal([self.state_num, neural_layer_1]))  # (2,8)
            # self.b1 = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)
            b1t = tf.Variable(tf.zeros([1, neural_layer_1]) + 0.1)

            l1t = tf.sigmoid(tf.matmul(self.state_input, w1t) + b1t)  # q_eval_input (?,2)

            w2t = tf.Variable(tf.random_normal([neural_layer_1, self.action_num]))
            # self.b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            b2t = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
            self.q_target = tf.matmul(l1t, w2t) + b2t

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        # with tf.variable_scope('soft_replacement'):
        # e赋值给t，当前网络的参数赋值给目标网络
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 取出当前动作的得分。
        # self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)

        self.loss = tf.reduce_mean(tf.square((self.q_target - self.q_eval)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # self.predict = tf.argmax(self.q_eval, 1)
        # print("w1:", w1.eval())
        # return (w1.eval(), b1.eval(), w2.eval(), b2.eval())

    def selectAction(self, currentState):
        """
        根据策略选择动作。
        :param currentState: 当前状态
        :return:
        """
        self.lock.acquire()
        if np.random.uniform() < self.epsilon:
            # 随机选择
            print("小概率随机选择动作")
            print("epsilon:",  self.epsilon)
            print("currentState: ", currentState)
            current_action_index = np.random.randint(0, self.action_num)
            print("action: ", current_action_index)
        else:
            print("通过神经网络选择动作")
            # 通过神经网络，输出各个actions的value值，选择最大value作为action。
            actions_value = self.session.run(self.q_eval, feed_dict={self.state_input: currentState})
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
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()  # 将头部弹出

        print("当前经验池数目：", len(self.replay_memory_store))
        self.memory_counter += 1
        self.lock.release()

    def experience_replay(self):
        """
        记忆回放。
        :return:
        """
        self.lock.acquire()
        print("------------进入经验回放！！！-----------")
        # 随机选择一小批记忆样本。
        # batch = self.BATCH if self.memory_counter > self.BATCH else self.memory_counter
        batch = int(self.OBSERVE / 2)
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
        q_next = self.session.run([self.q_target], feed_dict={self.state_input: batch_next_state})
        print("q_next: ", q_next)
        q_target = []
        for i in range(len(minibatch)):
            # 当前即时得分。
            current_reward = batch_reward[i][0]
            # print("batch_reward[i][0]: ", batch_reward[i][0])

            # # 游戏是否结束。
            # current_done = batch_done[i][0]

            # 更新 Q 值。
            # Q目标
            # q_value = current_reward + self.gamma * np.max(q_next[0][i])
            # print("q_next[0][i]:", q_next[0][i])
            # print("q_value:", q_value)

            # 当得分小于 0 时，表示走了不可走的位置。
            # if current_reward < 0:
            #     q_target.append(current_reward)
            # else:
            #     q_target.append(q_value)
            q_target.append(current_reward + self.gamma * np.max(q_next[0][i]))

        # print("q_target: ", q_target)
        q_target_new = np.expand_dims(q_target, axis=1)
        # print("q_target_new: ", q_target_new)

        # 变为和q_eval相同形状
        q_target_new1 = np.zeros((batch, self.B * self.A), dtype='float32')
        for i in range(batch):  # 行
            for j in range(2 * self.A):  # 列
                q_target_new1[i][j] = q_target_new[i][0]

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
                                   feed_dict={self.state_input: batch_state,
                                              self.q_target: q_target_new1})
        # print("learning_rate:", self.session.run(self.learning_rate))
        # learning_rate1 = self.session.run(self.learning_rate)
        # self.learning_rate_list.append(learning_rate1)
        # print("w1:", self.session.run(self.w1))
        # print("b1:", self.session.run(self.b1))
        # print("w2:", self.session.run(self.w2))
        # print("b2:", self.session.run(self.b2))

        self.cost_his.append(cost)
        self.lock.release()

    def step(self, currentState, actionIndex, agentSeq):
        """
        执行动作。
        :param agentSeq:
        :param currentState: 当前状态
        :param actionIndex:  当前动作的Index
        :return:
        """
        self.lock.acquire()
        self.theta = currentState[0][0]
        #  计算SNR
        SNR = ((self.H2 * self.theta * self.H1) * POWER[agentSeq]) / (
                ((sum(POWER) - POWER[agentSeq]) * self.interfereChannelGain) + (10 ** (-10))**2)

        # print("POWER[0] :", POWER[0] )
        # # print("sum(POWER) - POWER[0] :", sum(POWER) - POWER[0] )
        # print("POWER[agentSeq]:", POWER[agentSeq])
        # print("up:", (self.H2 * self.theta * self.H1) * POWER[agentSeq])
        # print("down:", float(
        #     ((sum(POWER) - POWER[agentSeq]) * self.interfereChannelGain) + 10 ** (-10)))

        print("SNR: ", SNR)
        #  计算速率
        RATE[agentSeq] = Band * math.log2(float(1 + SNR))
        print("Rate:", RATE)

        #  计算reward = 能量效率
        Psum = Pc + sum(POWER)
        ee = float(sum(RATE)) / Psum
        print("ee:", ee)

        # print("self.replay_memory_store[self.agentIndex[-1]][2]:", self.replay_memory_store[self.agentIndex[-1]][2])
        if len(self.agentIndex) > 0 and len(self.replay_memory_store) > 0:
            if self.step_index >= 1 and ee > list(self.replay_memory_store)[self.agentIndex[-1]][2]:
                reward = ee - self.replay_memory_store[-1][2]  # 当前ee大于上一个ee，正
            elif self.step_index >= 1 and ee < list(self.replay_memory_store)[self.agentIndex[-1]][2]:
                reward = ee - self.replay_memory_store[-1][2]  # 当前ee小于上一个ee，负
            else:
                reward = 0
        else:
            reward = 0

        print("当前的reward：", reward)
        nextState = np.reshape(self.action_list[actionIndex], (1, 2))
        if all(currentState , nextState):
            nextState = np.reshape(self.action_list[np.random.randint(0, self.action_num)], (1, 2))
            print("nextState: ", nextState)
        else:
           print("nextState: ", nextState)

        # self.cumulativeReward += (self.gamma ** self.step_index) * reward
        self.cumulativeReward += reward
        self.cumulativeList.append(self.cumulativeReward)
        print("self.cumulativeReward:" + "[" + str(agentSeq) + "] :", self.cumulativeReward)
        self.lock.release()

        return nextState, ee, reward, self.cumulativeReward

    def train(self, agentSeq):
        self.lock.acquire()
        print("self.action_list: \n", self.action_list)
        # 初始化当前状态。
        currentState = self.state_list

        # self.epsilon = self.INITIAL_EPSILON
        # print("\n第__" + str(agentSeq) + "__agent" + " start time: ", ctime())
        # print("\n第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练 Initial POWER:" + str(POWER) + "\n")
        reward_counter = 0
        self.agentIndex = []

        while True:
            print("\n第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练：")
            POWER[agentSeq] = currentState[0][1]
            # 选择动作。
            actionIndex = self.selectAction(currentState)
            # 执行动作，得到：下一个状态，执行动作的得分，是否结束。
            nextState, ee, reward, cumulativeReward = self.step(currentState, actionIndex, agentSeq)
            # 保存记忆。
            self.saveStore(currentState, actionIndex, ee, reward, nextState, cumulativeReward, agentSeq)
            # print("\nself.replay_memory_store[self.step_index][2]:", self.replay_memory_store[self.step_index][2])
            # print("\nself.replay_memory_store[self.step_index-1][2]:", self.replay_memory_store[self.step_index-1][2])

            # 每self.OBSERVE次，经验回放一次
            if self.step_index > 0 and self.step_index % self.OBSERVE == 0:
                # if self.step_index % self.OBSERVE == 0:
                self.experience_replay()

                # self.OBSERVE += self.step_index - self.OBSERVE
                # print("self.OBSERVE:", self.OBSERVE)

            #  每REPLACE_TARGET_FREQ次后，更新target网络
            if self.step_index % REPLACE_TARGET_FREQ == 0:
                print("更新一次target网络！！！")
                self.session.run(self.target_replace_op)

            # 在经验池中找到当前agent的数据的索引，即第几条数据是这个agent的
            for i, item in enumerate(list(self.replay_memory_store)):
                if item[-1] == agentSeq:
                    self.agentIndex.append(i)
            #  由于只需要最后一个值，前面的值可以删掉，防止列表溢出
                if self.step_index > 10 and len(self.agentIndex) > 10:
                    self.agentIndex.pop(0)

            #  如果reward 一直为0， reward_counter累加到一定值
            if reward == 0:
                reward_counter += 1
            else:
                reward_counter = 0
            #  如果循环了100次，将增大随机选择的概率
            # if reward_counter == 100:
            #     print("增大随机选择的概率!")
            #     self.INITIAL_EPSILON = 0.8

            # 训练停止条件
            # if 0 < list(self.replay_memory_store)[agentIndex[-1]][5] \
            #         - list(self.replay_memory_store)[agentIndex[-2]][5] < threshold:

            if len(self.agentIndex) > 0 and len(self.replay_memory_store) > 0:
                #  总的奖励大于10
                #  某一次的即时奖励大于10
                #  某个状态循环了100次
                print("判断停止条件！")
                print("reward_counter: ", reward_counter)
                if list(self.replay_memory_store)[self.agentIndex[-1]][2] >= 30\
                        or list(self.replay_memory_store)[self.agentIndex[-1]][5] >= 30 \
                        or reward_counter >= 500:
                    print("------------训练停止！！！！！！-----------")
                    print("reward_counter: ", reward_counter)
                    # print("replay_memory_store  \n     当前状态      |动作索引|奖励|   下个状态 ： ")
                    # for i in list(self.replay_memory_store):  # 转成list，否则deque迭代会产生错误
                    #     print(i, end=" ")
                    #     print("\n")
                    # print(str(list(self.replay_memory_store)[agentIndex[-1]][4]) +
                    #       "-" + str(list(self.replay_memory_store)[agentIndex[-2]][4]) + "<" + str(threshold))
                    # print("agentSeq:", agentSeq)
                    # print("agentIndex:", agentIndex)
                    # 转成list，否则deque迭代会出错
                    # 当cumulativeReward增加的值小于阈值，停止训练
                    EE.append((sum(RATE) / (sum(POWER) + Pc)))
                    print("第__" + str(agentSeq) + "__agent 训练完成\n" + "  POWER:"
                          + str(POWER[agentSeq]) + "," + "  角度:" + str(currentState[0][0])
                          + "," + "  EE:" + str(sum(RATE) / (sum(POWER) + Pc)) + "  POWER:" + str(
                        POWER) + "  RATE:" + str(RATE) + "\n")
                    # print("第__" + str(agentSeq) + "__agent 训练完成的 cost_his: ", self.cost_his)
                    global cost
                    cost.append(self.cost_his)
                    # self.plotCumulativeReward()
                    # print("cost:", cost)
                    #  数据清零
                    self.step_index = 0
                    # self.cumulativeReward = 0
                    # self.cost_his.clear()
                    # self.agentIndex.clear()
                    self.cumulativeReward = 0
                    # time.sleep(8)
                    # self.replay_memory_store.clear()
                    print("\n")
                    break  # 跳出while循环

            # 更新状态
            currentState = nextState
            # print("第" + str(agentSeq) + " agent "+"第" + str(self.step_index) + "次训练POWER:", POWER)
            print("第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练完POWER:", POWER)
            print("第__" + str(agentSeq) + "__agent " + "第" + str(self.step_index) + "次训练完角度:", currentState[0][0])
            self.step_index += 1

        self.lock.release()

    def compareWithRandom(self):
        #  随机的结果
        randomPower = []
        randomTheta = []
        rewardList = []
        # 取平均
        avergeReward = []
        # 多少次去平均
        avergeTime = 20
        for tim in range(1, times+1):
            print("第" + str(tim) + "次随机训练开始")
            random.seed(0)
            for i in range(M):
                randomPower.append(np.random.uniform(self.P_min, self.P_max))
                randomTheta.append(np.random.uniform(self.theta_min, self.theta_max))
            print("\nrandomPower:", randomPower)
            print("randomTheta:", randomTheta)

            SNR = [0] * M
            R = [0] * M

            for i in range(M):
                SNR[i] = float((self.H2 * randomTheta[i] * self.H1) * randomPower[i]) \
                         / (((sum(randomPower) - randomPower[i]) * self.interfereChannelGain) + (10 ** (-10))**2)
                R[i] = Band * math.log2(float(1 + SNR[i]))

            #  m个agent一次计算结束
            print("SNR: ", SNR)
            print("Rate:", R)
            Psum = Pc + sum(randomPower)
            rewardList.append(float(sum(R)) / Psum)
            print("rewardList:", rewardList)
            if tim % avergeTime == 0:
                avergeReward.append(float(sum(rewardList) / avergeTime))
                rewardList.clear()
            print("avergeReward:", avergeReward)
            print("第" + str(tim) + "次结束")
            randomPower.clear()
            randomTheta.clear()

        #  训练的结果
        EE_ = []
        global EE
        #  去掉相同元素
        # EE = list(set(EE))
        print("EE:", EE)
        #  最后训练完的才是系统的EE，取最后一个
        for item in range(len(EE)):
            if item % M == M - 1:
                EE_.append(EE[item])
        print("EE_:", EE_)
        # EE 是每个Agent训练结束时的能量效率，由于有延迟，以最后一个训练完的EE为准，即EE_
        # 将EE_取平均
        averageEE= []
        # 每avergeTime个元素之和
        s = 0
        for index in range(1, len(EE_) + 1):
            if index % avergeTime == 0:
                for m in range(index-avergeTime, index):
                    s += EE_[m]
                averageEE.append(float(s/avergeTime))

        #  画在同一张图上
        #  随机蓝色
        line1 = plt.plot(np.arange(len(avergeReward)), avergeReward, 'b-')
        #  训练红色
        line2 = plt.plot(np.arange(len(avergeReward)), EE_, 'r-')
        # plt.legend(handles=[line1, line2], labels=["随机训练结果", "DQN训练结果"], loc="upper right", fontsize=6)  # 图例
        plt.ylabel('EE')
        plt.xlabel('times')
        plt.show()
        EE.clear()
        avergeReward.clear()

    def plotCumulativeReward(self):
        #  画出CumulativeReward
        print("cumulativeList:", self.cumulativeList)
        plt.plot(np.arange(len(self.cumulativeList)), self.cumulativeList, 'g-')
        plt.ylabel('CumulativeReward')
        plt.xlabel('训练次数')
        plt.show()
        self.cumulativeList.clear()


def plot_cost():
    for i in range(M):
        plt.figure(i)
        plt.plot(np.arange(len(cost[i])), cost[i], 'b-')
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def plot_EE():
    global EE
    # EE = list(set(EE))
    print("EE:", EE)
    plt.plot(np.arange(len(EE)), EE, 'r-')
    plt.ylabel('次数')
    plt.xlabel('EE')
    plt.show()


if __name__ == '__main__':
    #  标志位
    threadOn = 0
    compareOn = 0
    #  多次训练画出EE
    EEWithTimes = 0

    if threadOn == 0 and compareOn == 0 and EEWithTimes == 1:
        for i in range(times):  # 多次训练
            print("第" + str(i) + "次训练！")
            #  实例化
            #  M为Agent个数
            virtualAgent = []
            for i in range(M):
                virtualAgent.append(VirtualAgent())
                # for j in range(100):
                # virtualAgent[i].compareWithRandom()
                # virtualAgent[i].train(i)

            threads = []
            t = []
            for i in range(M):
                t.append(threading.Thread(target=virtualAgent[i].train, args=(i,)))
                # t0 = threading.Thread(target=virtualAgent_0.train, args=(0,))
                # t1 = threading.Thread(target=virtualAgent_1.train, args=(1,))
                # t2 = threading.Thread(target=virtualAgent_2.train, args=(2,))

            # t1 = threading.Thread(target=virtualAgent_0.train(0))
            # t1 = threading.Thread(target=virtualAgent_0.train(), args=(0,))
            # 这样写无法实现多线程

            for i in range(M):
                threads.append(t[i])

            for item in threads:
                item.setDaemon(True)
                item.start()

            for item in threads:
                item.join()

            print("EE:", EE)
            print("第" + str(i) + "次训练结束！")

        #  子线程执行完毕，在主线程里画图
        plot_EE()
        EE.clear()

    if threadOn == 1 and compareOn == 0 and EEWithTimes == 0:
        # for tim in range(times):
        print("执行多线程！")
        #  实例化
        #  M为Agent个数
        virtualAgent = []
        threads = []
        t = []

        for i in range(M):
            virtualAgent.append(VirtualAgent())
            t.append(threading.Thread(target=virtualAgent[i].train, args=(i,)))

            # for j in range(100):
            # virtualAgent[i].compareWithRandom()
            # virtualAgent[i].train(i)
        # t0 = threading.Thread(target=virtualAgent_0.train, args=(0,))
        # t1 = threading.Thread(target=virtualAgent_1.train, args=(1,))
        # t2 = threading.Thread(target=virtualAgent_2.train, args=(2,))

        # t1 = threading.Thread(target=virtualAgent_0.train(0))
        # t1 = threading.Thread(target=virtualAgent_0.train(), args=(0,))
        # 这样写无法实现多线程

        for i in range(M):
            threads.append(t[i])

        for item in threads:
            item.setDaemon(True)
            item.start()

        for item in threads:
            item.join()

        # virtualAgent[0].plotCumulativeReward()
        # print("主线程cost:", cost)
        #  子线程执行完毕，在主线程里画图
        # plot_cost()
        cost.clear()

    if threadOn == 0 and compareOn == 0 and EEWithTimes == 0:
        print("不执行线程！")
        #  实例化
        #  M为Agent个数
        for tim in range(times):
            virtualAgent = []
            for i in range(M):
                virtualAgent.append(VirtualAgent())
                virtualAgent[i].train(i)
                virtualAgent[i].plotCumulativeReward()
            print("第" + str(tim) + "次训练结束！")

    if threadOn == 0 and compareOn == 1 and EEWithTimes == 0:
        print("不执行线程，之间比较！")
        #  实例化
        #  M为Agent个数
        virtualAgent = []
        for i in range(M):
            virtualAgent.append(VirtualAgent())
        virtualAgent[0].compareWithRandom()

    if threadOn == 1 and compareOn == 1 and EEWithTimes == 1:

        for tim in range(times):  # 多次训练
            print("开始DQN训练")
            print("第" + str(tim) + "次大的训练！")

            #  实例化
            #  M为Agent个数
            virtualAgent = []
            threads = []
            t = []
            for i in range(M):
                virtualAgent.append(VirtualAgent())
                # for j in range(100):
                # virtualAgent[i].compareWithRandom()
                # virtualAgent[i].train(i)

                t.append(threading.Thread(target=virtualAgent[i].train, args=(i,)))
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

            print("EE:", EE)
            print("第" + str(tim) + "次训练结束！")

        print("结束DQN训练")

        print("开始随机计算")
        #  实例化
        #  M为Agent个数
        virtualAgent = []
        for i in range(M):
            virtualAgent.append(VirtualAgent())
        virtualAgent[0].compareWithRandom()
