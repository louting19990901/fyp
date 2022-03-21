import gym
from py_client_sb import YardEnv
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
from stable_baselines3 import PPO, A2C

# env = YardEnv(16, 10011, 'train')
# episode=100
# total_task_number=210  #average step per episode
# model = PPO("MlpPolicy", env, verbose=1).learn(total_task_number*episode)
# obs = env.reset()
# while True:
#   action, _ = model.predict(obs, deterministic=True)
#   obs, reward, done, info = env.step(action)
#   env.render()
#   print("testing")
#   if done:
#     print("Goal reached!", "reward=", reward)
#     break



def make_env( rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init(env_id):
        env = YardEnv(16, env_id, 'train')
        # env_id+=1
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

# env_id = 'CartPole-v1'
# The different number of processes that will be used
PROCESSES_TO_TEST = [1, 2, 4, 8, 16]
NUM_EXPERIMENTS = 3 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)

episode=30
total_task_number=210  #average step per episode
TRAIN_STEPS = total_task_number*episode
# Number of episodes for evaluation
EVAL_EPS = 20
ALGO = A2C

# We will create one environment to evaluate the agent on
eval_env =  YardEnv(16, 0, 'train')
# eval_env.reset()

reward_averages = []
reward_std = []
training_times = []
total_procs = 0


if __name__=="__main__":

    for n_procs in PROCESSES_TO_TEST:
        port_id=1
        total_procs += n_procs
        print('Running for n_procs = {}'.format(n_procs))
        if n_procs == 1:
            # if there is only one process, there is no need to use multiprocessing
            train_env = DummyVecEnv([lambda: YardEnv(16,1, 'train')])
        else:
            print("2222 ",n_procs)
            # Here we use the "fork" method for launching the processes, more information is available in the doc
            # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
            train_env = SubprocVecEnv([make_env(i+2, i+total_procs) for i in range(n_procs)], start_method='spawn')




        rewards = []
        times = []

        for experiment in range(NUM_EXPERIMENTS):
            # it is recommended to run several experiments due to variability in results
            train_env.reset()
            model = ALGO('MlpPolicy', train_env, verbose=0)
            start = time.time()
            model.learn(total_timesteps=TRAIN_STEPS)
            times.append(time.time() - start)
            mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
            rewards.append(mean_reward)
        # Important: when using subprocess, don't forget to close them
        # otherwise, you may have memory issues when running a lot of experiments
        train_env.close()
        reward_averages.append(np.mean(rewards))
        reward_std.append(np.std(rewards))
        training_times.append(np.mean(times))

    training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2)
    plt.xlabel('Processes')
    plt.ylabel('Average return')
    plt.subplot(1, 2, 2)
    plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
    plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
    plt.xlabel('Processes')
    _ = plt.ylabel('Training steps per second')
