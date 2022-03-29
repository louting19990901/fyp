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

if __name__ == '__main__':
    # env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    print("111111")

    # env=DummyVecEnv([YardEnv(16,10,"train"),YardEnv(16,11,"train"),YardEnv(16,12,"train"),YardEnv(16,13,"train")])
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    eval_env = YardEnv(16, 50, 'test')
    episode = 10000
    total_task_number = 200

    core=4


    env = SubprocVecEnv([make_env(i + 10, i + 10) for i in range(core)])
    tic = time.time()
    # model = PPO('MlpPolicy', env, verbose=1)
    # model = A2C('MlpPolicy', env, verbose=1)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_task_number*episode)
    toc = time.time()
    due = toc - tic
    print("==========")
    print(core," core take ",due)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print()
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    #plot





    #
    #
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()