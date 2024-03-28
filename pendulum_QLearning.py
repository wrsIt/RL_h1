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

    def update_env(self, observation, episode, step_counter):
        if is_final(observation):

    def train(self, env, ):
        q_table = build_q_table()
        for episode in range(self.MAX_EPISODES):
            step_counter = 0
            # 与环境交互
            observation, _ = self.env.reset()
            observation = tuple(observation)

            # 单个回合训练
            is_terminated = False  # 标记该回合是否结束
            update_env(observation, episode, step_counter)  # 更新环境
            while step_counter < self.one_turn_step and (not is_terminated):
                # 离散当前状态 以便在q表中索引
                observation_round = round_observation(observation)
                max_Q_action = choose_max_Q_action(observation_round, self.q_table)
