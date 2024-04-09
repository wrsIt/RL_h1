import math
import os

import numpy as np
import pandas
import pandas as pd

from pendulum_env import PendulumEnv


# 用numpy加速计算
class Agent:
    def __init__(self, theta_n=60, d_theta_n=40, action_n=3, gamma=0.98):
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

        # 初始化Q(state, action)表，其中state=(theta, d_theta)
        self.MAX_EPISODES = 100  # 智能体在环境中运行的序列的数量
        self.EPSILON = 0.9  # 贪心策略的参数
        self.alpha = 0.1  # 更新时的学习率
        self.lam = 0.98  # 折扣因子
        self.ts = 0.005  # 刷新时间
        self.one_turn_step = 50000

        self.data_dir = "./result/%s_th-%s_dth-%s_a-%s/" % (
            self.model_name, self.theta_n, self.d_theta_n, self.action_n)  # 数据保存路径
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def build_q_table(self, n_angle_states, n_angular_velocity_states, u_actions):
        index = pd.MultiIndex.from_product([n_angle_states, n_angular_velocity_states])  # 以(角度，角速度)为索引构建Q表
        columns = u_actions  # 列值为动作集
        # 构造Q表
        table = pd.DataFrame(
            np.zeros((len(n_angle_states) * len(n_angular_velocity_states), len(u_actions))), index=index,
            columns=columns
        )
        table.index.names = ['theta', 'd_theta']  # 标注索引名
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
        theta = self.discretize_mapping(x_, self.THETA_STATES)
        d_theta = self.discretize_mapping(d_theta, self.D_THETA_STATES)
        return theta, d_theta  # 单位是弧度

    # def update_env(self, state, episode, step_counter):
    #     if self.is_final_state(state):
    #         content = "Episode %s : total step = %s" % (episode + 1, step_counter)
    #         print('\n', content)
    #     else:
    #         state = self.discretize_state(state)  # 离散化状态 弧度
    #         content = "Episode %s：step = %s" % (episode + 1, step_counter)
    #         content += " angle：" + str(state[0] * 180 / math.pi) + ", angular_velocity：" + str(
    #             state[1] / math.pi) + "pi"
    #         print('\n', content)
    def update_env(self, state, episode, step_counter):
        if self.is_final_state(state):
            interaction = "Episode %s：total_step = %s" % (episode + 1, step_counter)
            print('\r{}'.format(interaction), end='')
            # time.sleep(2)
            print('\n wowow finished!')
        else:
            state = self.discretize_state(state)
            interaction = "Episode %s：step = %s" % (episode + 1, step_counter)
            interaction += " angle：" + str(state[0] * 180 / math.pi) + ", angular_velocity：" + str(
                state[1] / math.pi) + "pi"
            print('\r{}'.format(interaction), end='')

    def choose_max_Q_action(self, state: tuple, table: pandas.DataFrame):
        state_actions = table.loc[state]  # 这个状态对应的一行
        if (np.random.random() > self.EPSILON) or (state_actions.sum() == 0):
            action = np.random.choice(self.actions)  # ε-greedy中探索性选择动作
        else:
            action = state_actions.idxmax()  # 贪心策略：选概率较大的动作(返回具有最大值的索引位置)
        return action

    def reward(self, th, thdot, u):
        costs = 5 * (angle_normalize(th) ** 2) + 0.1 * (thdot) ** 2 + u ** 2  # cost是reward的相反数，是正的。reward是负的
        return -costs

    def get_d_d_theta(self, theta, d_theta, u):
        m = 0.055
        g = 9.81
        l = 0.042  # 重心到转子的距离
        J = 1.91e-4  # 转动惯量
        b = 3e-6  # 粘滞阻尼
        K = 0.0536  # 转矩常数
        R = 9.5
        theta = angle_normalize(theta)  # ************要加这一步
        d_d_theta = (1 / J) * (
                m * g * l * math.sin(theta) - b * d_theta - (K ** 2 / R) * d_theta + (
                K / R) * u)
        return d_d_theta

    def get_feedback(self, state, action):
        theta = state[0]
        d_theta = state[1]  # 角速度/pi
        reward = self.reward(theta, d_theta, action)
        # print(theta,d_theta,action,reward)
        d_d_theta = self.get_d_d_theta(theta, d_theta, action)  # 有点问题

        # 更新角速度
        d_theta_new = d_theta + self.ts * d_d_theta
        d_theta_new = np.clip(d_theta_new, min(self.D_THETA_STATES), max(self.D_THETA_STATES))

        # 更新角度
        theta_new = theta + self.ts * d_theta_new
        new_state = (theta_new, d_theta_new)
        return new_state, reward

    def get_feedback_env(self, env, action):
        action = [action]
        state, reward = env.step(action)
        state = tuple(state)
        return state, reward

    def train(self, env, is_view=False):
        q_table = self.build_q_table(self.THETA_STATES, self.D_THETA_STATES, self.actions)
        for episode in range(self.MAX_EPISODES):
            step_counter = 0
            # 与环境交互
            if is_view:
                observation, _ = env.reset()
                observation = tuple(observation)
            else:
                observation, _ = env.reset(is_view=is_view)
                observation = tuple(observation)
            # observation, _ = env.reset(is_view=is_view)
            # observation = tuple(observation)
            # 单个回合训练
            is_terminated = False  # 标记该回合是否结束
            self.update_env(observation, episode, step_counter)  # 更新环境
            while step_counter < self.one_turn_step and (not is_terminated):
                # 离散当前状态 以便在q表中索引
                observation_discretized = self.discretize_state(observation)
                max_Q_action = self.choose_max_Q_action(observation_discretized, q_table)
                max_Q = q_table.loc[observation_discretized, max_Q_action]
                if not is_view:
                    observation_new, reward = self.get_feedback(observation, max_Q_action)
                else:
                    observation_new, reward = self.get_feedback_env(env, max_Q_action)

                # Q-target
                if not self.is_final_state(observation_new):
                    q_target = reward + self.lam * max(q_table.loc[self.discretize_state(observation_new)])
                else:
                    q_target = reward
                    self.update_env(observation, episode, step_counter)
                    is_terminated = True
                #  更新q表
                q_table.loc[observation_discretized, max_Q_action] += self.alpha * (q_target - max_Q)
                observation = observation_new
                self.update_env(observation, episode, step_counter)
                # print("!!!!!!!!!", observation, q_target)
                step_counter += 1
        return q_table

    def read_q_table(self, env):
        self.q_table = pd.read_csv('data/result_q_table.csv', index_col=[0, 1], header=0)
        is_view = True
        state, _ = env.reset(is_view=is_view)
        state = tuple(state)  # 换一下形式
        step_counter = 0
        is_terminated = False
        while step_counter < self.one_turn_step and (not is_terminated):
            state_round = self.discretize_state(state)
            # 拿到最大Q值的动作
            action = self.choose_max_Q_action(state_round, self.q_table)
            action = np.array([action], dtype='float64')
            next_state, reward = env.step(action)
            if self.is_final_state(next_state):
                is_terminated = True
                env.close()

            state = next_state
            step_counter += 1

        print(self.q_table)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == "__main__":
    env = PendulumEnv("human")
    agent = Agent()
    # q_table = agent.train(env, is_view=False)
    # q_table.to_csv('data/result_q_table.csv', index=True, header=True, date_format='%.4f')
    agent.read_q_table(env)
