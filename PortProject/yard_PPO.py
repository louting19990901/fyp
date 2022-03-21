import gym
from py_client_sb import YardEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


env = YardEnv(16, 0, 'train')
episode=1
total_task_number=210  #average step per episode

model = PPO("MlpPolicy", env, verbose=1).learn(total_task_number*episode)


print("end training")

#plot
# plt.xticks(range(len(env.episode_list)))
# plt.plot(env.episode_list, env.relocation_list)

# x=[2,4,6,8]
# y=[1,2,3,4]
# plt.plot(x,y)

# print("111",env.episode_list, env.relocation_list)
# plt.plot(env.episode_list, env.relocation_list)
# plt.show()
model.save("ppo_yard")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_yard")

obs = env.reset()

while True:

  action, _ = model.predict(obs, deterministic=True)
  print("action is ",action)
  obs, reward, done, info = env.step(action)
  env.render()

  print(obs,reward,done,info)
  print("testing")
  if done:
    print("Goal reached!", "reward=", reward)
    break