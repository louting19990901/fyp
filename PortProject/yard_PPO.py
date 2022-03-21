import gym
from py_client_sb import YardEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

env = YardEnv(16, 10011, 'train')
episode=1000
total_task_number=210  #average step per episode

model = PPO("MlpPolicy", env, verbose=1).learn(total_task_number*2500)


print("end training")
model.save("ppo_yard")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_yard")

obs = env.reset()

while True:

  action, _ = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)
  env.render()
  print("testing")
  if done:
    print("Goal reached!", "reward=", reward)
    break