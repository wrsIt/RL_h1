import math
import os

import numpy as np
import pandas as pd

from pendulum_env import PendulumEnv


# 用numpy加速计算
class Agent:
    def __init__(self, theta_n=200, d_theta_n=200, action_n=3, gamma=0.98):
        self.model_name = self.__class__.__name__  # 模型名称

        self.gamma = gamma  # 衰减因子
        self.theta_n = theta_n  # 角度划分为多少份
        self.d_theta_n = d_theta_n  # 角速度划分为多少份
        self.action_n = action_n  # 本题中的action是电压u

        self.max_action = 3.  # 最大动作（电压）

        self.actions = np.linspace(-self.max_action, self.max_action, self.action_n)  # 动作空间 -3 0 3
        self.env = PendulumEnv()  # 交互环境
        self.THETA_STATES = np.round(np.linspace(-math.pi, math.pi, self.theta_n), decimals=4)
        self.D_THETA_STATES = np.round(np.linspace(-15 * math.pi, 15 * math.pi, self.d_theta_n), decimals=4)

        self.q_table = np.zeros((self.theta_n, self.d_theta_n, len(self.actions)))
        # 初始化Q(state, action)表，其中state=(theta, d_theta)
        self.MAX_EPISODES = 100  # 智能体在环境中运行的序列的数量
        self.EPSILON = 0.9  # 贪心策略的参数
        self.alpha = 0.1  # 更新时的学习率
        self.lam = 0.98  # 折扣因子
        self.ts = 0.0001  # 刷新时间
        self.one_turn_step = 50000

        self.data_dir = "./result/%s_th-%s_dth-%s_a-%s/" % (
            self.model_name, self.theta_n, self.d_theta_n, self.action_n)  # 数据保存路径
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def build_q_table(self, n_angle_states, n_angular_velocity_states, u_voltage_actions):
        index = pd.MultiIndex.from_product([n_angle_states, n_angular_velocity_states])  # 以(角度，角速度)为索引构建Q表
        columns = u_voltage_actions  # 列值为动作集

        # 构造Q表
        table = pd.DataFrame(
            np.zeros((len(n_angle_states) * len(n_angular_velocity_states), len(u_voltage_actions))), index=index,
            columns=columns
        )
        table.index.names = ['angle', 'angular_velocity']  # 标注索引名
        # print(table)
        return table

    def is_final_state(self, state):
        # final state:将摆杆摆起并稳定 state:(0,0) 状态区间是对称的，所以稳定状态就是当角度、角速度分别处于各自状态区间的中间位置时。
        theta = state[0]
        d_theta = state[1]
        return self.is_mid_state(theta, self.THETA_STATES) and self.is_mid_state(d_theta, self.D_THETA_STATES)

    def is_mid_state(self, t, states):
        # 根据给定的状态列表，判断给定的值是否处于这个状态列表的中间位置
        idx_mid = len(states) // 2
        if len(states) % 2 == 0:
            right = states[idx_mid]
            left = states[idx_mid - 1]
            return left < t < right
        else:
            mid = states[idx_mid]
            return t == mid

    def discretize_mapping(self, theta, states):
        # theta是已经映射到[-pi,pi]的角度
        # 查找theta在列表中该在哪个分组，返回离散化后对应的数值
        idx = np.digitize(theta, bins=states)
        if 0 < idx < len(states):
            left = states[idx - 1]
            right = states[idx]
            # 看哪边更接近theta
            if np.abs(left - theta) < np.abs(right - theta):
                idx = idx - 1  # 左边更接近
            else:
                idx = idx
        elif idx <= 0:
            idx = 0
        else:
            idx = len(states) - 1
        return states[idx]

    def discretize_state(self, state):
        # 离散化状态 切分区间->区间对应的状态数据
        theta = state[0]
        d_theta = state[1]
        theta_copy = theta
        # 角度映射到[-pi,pi] fmod是取余，参数是2pi，由于fmod函数的特性导致其范围是[0,2pi]，最后减去pi来平移区间。
        x_ = np.fmod(theta_copy + np.pi, 2 * np.pi) - np.pi
        theta = self.discretize_mapping(theta, self.THETA_STATES)
        d_theta = self.discretize_mapping(d_theta, self.D_THETA_STATES)
        return theta, d_theta  # 单位是弧度

    def update_env(self, state, episode, step_counter):
        if self.is_final_state(state):
            content = "Episode %s : total step = %s" % (episode + 1, step_counter)
            print('\n', content)
        else:
            state = self.discretize_state(state)  # 离散化状态 弧度
            content = "Episode %s：step = %s" % (episode + 1, step_counter)
            content += " angle：" + str(state[0] * 180 / math.pi) + ", angular_velocity：" + str(
                state[1] / math.pi) + "pi"
            print('\n', content)

    def choose_max_Q_action(self, state, q_table):
        state_actions = q_table.loc[state]  # 这个状态对应的一行


        def train(self, env, is_view=False):
            q_table = build_q_table()
            for episode in range(self.MAX_EPISODES):
                step_counter = 0
                # 与环境交互
                observation, _ = env.reset(is_view=is_view)
                observation = tuple(observation)
                # 单个回合训练
                is_terminated = False  # 标记该回合是否结束
                self.update_env(observation, episode, step_counter)  # 更新环境
                while step_counter < self.one_turn_step and (not is_terminated):
                    # 离散当前状态 以便在q表中索引
                    observation_discretized = self.discretize_state(observation)
                    max_Q_action = self.choose_max_Q_action(observation_discretized, self.q_table)
