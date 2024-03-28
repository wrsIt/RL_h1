import gym

env = gym.make('Tennis-v4', render_mode="human")
print("Observation Space: ", env.observation_space)
print("Action Space       ", env.action_space)

obs = env.reset()
for i in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info, _ = env.step(action)
env.close()

print(gym.envs.registry.keys())
