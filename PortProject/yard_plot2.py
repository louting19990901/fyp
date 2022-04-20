import gym
# from py_client_sb import YardEnv
from py_client_v2 import YardEnv
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

global_relocation_list=[]
test_relocation_list=[]

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = YardEnv(16,env_id,"train",global_relocation_list)
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
    episode = 2000
    total_task_number = 200

    # envs=[make_env(i + 20, i + 20) for i in range(num_cpu)]

    env=YardEnv(16,29,'train',global_relocation_list)
    # env = SubprocVecEnv(envs)
    model = A2C('MlpPolicy', env, verbose=1,tensorboard_log="./yar"
                                                            "d_tensorboard/")

    tic = time.time()
    model.learn(total_timesteps=total_task_number*episode)
    toc = time.time()
    due = toc - tic
    print(num_cpu," core take ",due)


    # print(global_relocation_list)
    # eval_env = YardEnv(16, 50, 'test',test_relocation_list)
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



    print("total mean ",getMean(global_relocation_list))
    print("best min index is ",env.bestMeanIndex)
    print("best min is ",env.bestMean)
    # print("last 100 mean ",getMean(global_relocation_list[-100:]))
    plt.plot( range(len(global_relocation_list)),global_relocation_list)
    plt.show()



