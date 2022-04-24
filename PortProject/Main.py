import gym
# from py_client_v1 import YardEnv
# from py_client_v2 import YardEnv
# from py_client_v3 import YardEnv
from py_client_v4 import YardEnv
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

def getMean(arr):
    sum=0.0
    for i in arr:
        sum+=i
    return float(sum)/float(len(arr))

if __name__ == '__main__':
    episode = 2000
    total_task_number = 200

    env=YardEnv(16,29,'train',global_relocation_list)
    # env = SubprocVecEnv(envs)
    model = A2C('MlpPolicy', env, verbose=1,tensorboard_log="./yar"
                                                            "d_tensorboard/")

    tic = time.time()
    model.learn(total_timesteps=total_task_number*episode)
    toc = time.time()
    due = toc - tic

    print("total mean ",getMean(global_relocation_list))
    print("The episode of Min Average NAR is ",env.bestMeanIndex)
    print("Min Average NAR is ",env.bestMean)
    plt.plot( range(len(global_relocation_list)),global_relocation_list)
    plt.show()



