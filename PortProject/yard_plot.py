import gym
# from py_client_sb import YardEnv
from py_client_reward import YardEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from stable_baselines3.common.evaluation import evaluate_policy
import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C,DQN

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = YardEnv(16,env_id,"train")
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def getMean(arr):
    sum=0.0
    for i in arr:
        sum+=i
    return float(sum)/float(len(arr))

if __name__ == '__main__':
    # env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    eval_env = YardEnv(16, 40, 'test')
    episode = 100
    total_task_number = 200
    core=4

    env = SubprocVecEnv([make_env(i + 20, i + 20) for i in range(core)])
    model = PPO('MlpPolicy', env, verbose=1)

    tic = time.time()
    model.learn(total_timesteps=total_task_number*episode)
    toc = time.time()
    due = toc - tic
    print(core," core take ",due)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    print(getMean(eval_env.relocation_list))
    plt.plot( range(len(eval_env.relocation_list)),eval_env.relocation_list)
    plt.show()


    #
    #
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()