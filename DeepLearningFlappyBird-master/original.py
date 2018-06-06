#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game    # 导入游戏环境
import random
import numpy as np
from collections import deque   # 从collection模块引入双端队列deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # 能够输出的动作数量（向上或者下降）
GAMMA = 0.99 # 对于过去观测值得衰减指数


REPLAY_MEMORY = 50000 # 记忆矩阵的容量大小（行数）
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

# 初始训练的值
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1   # 以epsilon的概率随机选择动作，以1-epsilon的概率选择有最大q值的action
'''
# 最优化训练后的值
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
'''


def weight_variable(shape):    # 定义用于创建神经网络的权重变量weights的函数
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)


def bias_variable(shape):       # 定义用于创建神经网络的偏置bias的函数
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


# 调用tensorflow的实现卷积的函数tf.nn.conv2d
def conv2d(x, W, stride):
    # 第一个参数是输入input；第二个参数是卷积核; 第三个参数的步长； 第四个参数用于决定不同的卷积方式
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")   # 结果是返回一个tensor


# 定义max_pooling池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


# 构造神经网络，填入各个权重、偏置的参数
def createNetwork():
    # network weights

    # 第一个卷积层的参数
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    # 第二个卷积层的参数
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    # 第三个卷积层的参数
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    # 第一个全连接层的参数
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    # 第二个全连接层的参数
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # 原始输入层的palceholder
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 构造第一个卷积层，调用了conv2d函数，输入为s
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)   #卷积层
    h_pool1 = max_pool_2x2(h_conv1)     # 池化层

    # 构造第二个卷积层，输入为上层的h_pool1
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    # 构造第三个卷积层，输入为上层的h_conv2
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    """reshape最后一个卷积层的输出的列数来匹配全连接层节点数"""
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    # reshape 这里的 [-1, 1600] 表示变成行数未知（根据列数计算），列数为1600
    # 因为后面要进行矩阵的运算， 所以这里的列数要和后面的 W_fc1 匹配


    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # output layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


# 训练神经网络
# s_t是每轮的第一个状态，s_t1是当前状态，每一step介绍即替换状态，准备下一轮step
def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    # 开启与游戏模拟器环境的交互
    game_state = game.GameState()

    # store the previous observations in replay memory
    # 用于存储过去的observation的双端队列
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)  # action数目为2，所以这里返回[0，0]
    do_nothing[0] = 1               # 只有两个动作（上升和不做动作），不做动作对应数组的第二个元素，令其为1激活此动作
    x_t, r_0, terminal = game_state.frame_step(do_nothing)  # do_nothing动作输入到step(),返回观察值，reward，和是否结束
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)      # resize尺寸，并转化为灰度图, 返回的图片为单通道
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)   # 二值化图片， 阈值为1，填充色为255, x_t为转化后的结果

    # 将返回的图片x_t拼凑四张成为第一个state（s_t）
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # stack：对arrays里面的每个元素(可能是个列表，元组，或者是个numpy的数组)变成numpy的数组后，再对每个元素增加一维(至于维度加在哪里，是靠axis控制的)

    # 用于存储和加载网络训练参数
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 训练循环的入口
    epsilon = INITIAL_EPSILON
    t = 0  # 初始化time step
    while "flappy bird" != "angry bird":
        # 使用epsilon greedy的方法选择动作
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]  # 把当前的初始状态s_t输入神经网络得出action的q值
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:   # FRAME_PER_ACTION参数控制每一帧有多少个动作，这里等于一，所以每个time step选一次动作
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:       #如果FRAME_PER_ACTION不为一的话，那么周期内其他时间的timestep默认做down动作
            a_t[0] = 1 # do nothing

        # 按照比例缩减epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        # 执行以上程序选出的动作a_t， 然后观察环境返回的下一状态state（imagedata）和获得的奖励reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)  # 返回的下一状态image也要灰度转换
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)      #二值化转换
        x_t1 = np.reshape(x_t1, (80, 80, 1))    # 重新reshape成需要的尺寸
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)       # 这里是下一状态s_t1

        # store the transition in D
        # 存储transition于队列D中，从右边开始
        # 如果队列的长度大于预定的size， 即从左边开始弹出旧的元素
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        # 只有当step 大于 OBSERVE，才开始训练神经网络，否则，只是把transition存储于replay_memory中
        if t > OBSERVE:
            # sample a minibatch to train on
            # 从队列中抽样一个mini-batch用于梯度下降
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            # 根据minibatch里面的不同属性的索引抽取出需要的属性
            s_j_batch = [d[0] for d in minibatch]   # d[0]对应的为state
            a_batch = [d[1] for d in minibatch]     # d[1]对应的是action
            r_batch = [d[2] for d in minibatch]     # d[2]对应的是reward
            s_j1_batch = [d[3] for d in minibatch]  # d[3]对应的是下一个state


            # mini batch 梯度下降训练
            # y_batch 用于存储计算出来的q值， 作为target， 用于计算loss，进而梯度下降
            # 提取mini batch存到y_batch中
            y_batch = []
            # 把上面获得的 s_j1_batch（下一个state）扔回神经网络，获得下一个动作的q值
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})  # eval()执行括号内的表达式并返回值
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]      # minibatch的[4]位置对应terminal
                # if terminal, only equals reward
                if terminal:        # 如果结束了，就只返回reward，因为没有下一个action和state了
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))    #没结束的话，照常计算q值
            # perform gradient step
            # 根据抽样得出的batch数据进行训练
            train_step.run(feed_dict = {
                y : y_batch,   # 由上面计算出来的目标y值
                a : a_batch,    # sample出来的action值
                s : s_j_batch}  # sample出来的状态值
            )
        # update the old values
        s_t = s_t1   # 把当前的状态赋值到旧状态，准备下一轮step循环
        t += 1      # 进入下一轮的step

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # 于命令行打印状态信息
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''
# 构造主函数
def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
