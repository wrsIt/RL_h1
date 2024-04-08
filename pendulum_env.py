__credits__ = ["Carlos Luis"]

import math
from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumEnv(gym.Env):
    """
    ## Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)

    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ## Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ## Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 200 time steps.

    ## Arguments

    - `g`: .

    Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
    the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
    The default value is `g = 10.0`.
    On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.

    ## Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81):
        self.max_theta = np.pi  # 角度
        self.max_speed = 15 * math.pi  # 角速度
        self.max_voltage = 3.0  # 扭矩，控制动作（电压）u
        self.dt = 0.005  # 采样时间
        self.g = g
        self.m = 0.055
        self.l = 0.042  # 重心到转子的距离
        self.J = 1.91e-4  # 转动惯量
        self.b = 3e-6  # 粘滞阻尼
        self.K = 0.0536  # 转矩常数
        self.R = 9.5
        self.state = np.array([-math.pi, 0])

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        # high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        high = np.array([self.max_theta, self.max_speed])
        self.action_space = spaces.Box(
            low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def get_state(self):  # 获取现在的状态
        return self.state

    def reward(self, th, thdot, u):
        costs = 5 * angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + u ** 2  # cost是reward的相反数，是正的。reward是负的
        return -costs

    def get_d_d_theta(self, theta, d_theta, u):
        m = self.m
        g = self.g
        l = self.l
        b = self.b
        K = self.K
        R = self.R
        J = self.J

        d_d_theta = (1 / J) * (m * g * l * math.sin(theta) - b * d_theta - (K ** 2 / R) * d_theta + (K / R) * u)
        return d_d_theta

    def new_state(self, theta, d_theta, u):
        # 计算新状态
        dt = self.dt
        d_d_theta = self.get_d_d_theta(theta, d_theta, u)
        new_theta = theta + dt * d_theta
        new_d_theta = d_theta + dt * d_d_theta
        # 换一下形式
        new_theta = angle_normalize(new_theta)
        new_d_theta = np.clip(new_d_theta, -self.max_speed, self.max_speed)
        return new_theta, new_d_theta


    def step(self, u):
        th, thdot = self.state  # th，theta，角度；thdot角速度

        dt = self.dt
        u = np.clip(u, -self.max_voltage, self.max_voltage)[0]
        self.last_u = u  # for rendering

        reward = self.reward(th, thdot, u)
        new_theta, new_d_theta = self.new_state(th, thdot, u)
        self.state = np.array([new_theta, new_d_theta])  # 更新状态

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        # 返回状态和reward
        return self._get_obs(), reward

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        return np.array([theta, thetadot], dtype=np.float32)

    def render(self):  # 渲染图像
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def action_sample(self):  # 把0，1，2策略映射成-3,0,3
        action_map = {0: np.array([-3.]), 1: np.array([0.]), 2: np.array([3.])}
        action = self.action_space.sample()
        action = action_map[action]
        return action

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
