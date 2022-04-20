import socket
import jpype
import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gym
import torch.nn.functional as F
import math

import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env

show_info = False


class Observation:  # total 174 bit => total 30
    bay = -1  # 1 bit
    stack = -1  # 1 bit
    containersMatrix = ""  # 6*25 =150 bit =>6
    headingTrucksNumber = -1  # 1 bit
    queuingTrucksNumber = -1  # 1 bit
    headingContainers = ""  # 10 bit
    queuingContainers = ""  # 10 bit
    relocationNumber = -1

    def __init__(self, bay, stack, containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers,
                 queuingContainers, relocationNumber):
        self.stack = stack
        self.bay = bay
        self.containersMatrix = containersMatrix
        self.headingTrucksNumber = headingTrucksNumber
        self.queuingTrucksNumber = queuingTrucksNumber
        self.headingContainers = headingContainers
        self.queuingContainers = queuingContainers
        self.relocationNumber = relocationNumber

def getMean(arr):
    sum=0.0
    for i in arr:
        sum+=i
    return float(sum)/float(len(arr))

class YardEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    observation = None
    count = 0
    zero_port = 10006
    is_connected = False
    relocation_list = []
    episode_list = []
    last100relocationNumbers = np.zeros(100)
    last100Rewards = np.zeros(100)
    global_relocation_list=[]
    additionalRelocation=0
    bestMean=99999
    bestMeanIndex=-1

    def __init__(self, n_actions, port, env_type,global_relocation_list):
        super(YardEnv, self).__init__()
        self.client = None
        # print("ini port is ",port)
        self.port = port
        self.is_connected = False

        self.qc_num = n_actions
        self.env_type = env_type
        self.root_path = os.getcwd()
        self.executor = self.start_java_end()
        self.global_relocation_list=global_relocation_list
        self.action_space = spaces.Discrete(6)
        # Example for using image as input (channel-first; channel-last also works):

        self.observation_space = spaces.Box(low=0, high=25,
                                            shape=(30,), dtype=np.uint8)

        # start server and client

    def start_java_end(self):
        # print("self.port: ",self.port)
        # print("1: ",os.getcwd(),type(os.getcwd()))

        # if (os.getcwd() != r"C:\Users\86189\Desktop\fyp\fyp\PortProject"):
        #     os.chdir("..")
        # os.chdir("..")
        # print("2: ", os.getcwd())
        os.chdir("C:/Users/86189/Desktop/fyp/fyp/PortProject/JavaProject/bin")

        # print("3: ", os.getcwd())

        jvmPath = jpype.getDefaultJVMPath()
        jar_path = '-Djava.class.path={}'.format(
            self.get_jars(self.root_path))
        if not jpype.isJVMStarted():
            jpype.startJVM(jvmPath, jar_path)
        # print("4:",os.getcwd())
        javaClass = jpype.JClass("Environment.Executor")

        '''
        # constructor
        #         javaInstance = javaClass(1,2)
        # 
        #         # att ributes
        #         print(javaInstance.a)
        #         print(javaInstance.b)	
        # 
        #         # invoke function
        result = javaInstance.test_func(123)
        print(result)
        '''
        # print("5: ", os.getcwd())
        os.chdir("..")
        # print("6: ", os.getcwd())
        # print("current port: ",self.port+self.zero_port)
        executor = javaClass(self.zero_port + self.port, self.env_type)  # get an instance of java object

        # javaInstance = javaClass.getInstance('demo')  # invoke class static function directly

        # jpype.shutdownJVM()
        '''
        OSError: JVM cannot be restarted
        jpype.startJVM(jvmPath)
        print(jpype.isJVMStarted())
        '''
        return executor

    def get_jars(self, path):
        results = []
        filter = ['.jar']
        for maindir, subdir, files in os.walk(path):
            for f in files:
                apath = os.path.join(maindir, f)
                ext = os.path.splitext(apath)
                if ext[1] in filter:
                    results.append(apath)
        temp = ""
        for i in range(len(results)):
            temp = temp + results[i] + ";"
        return temp

    def reset(self):

        # start java server and simulation
        self.executor.startServer()

        # connect to server
        # print("port ",self.port," connect: ",self.is_connected)
        if not self.is_connected:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect(('127.0.0.1', self.zero_port + self.port))
            self.is_connected = True

        info = json.loads(str(self.client.recv(1024), encoding="GBK"))

        # -----------state-----------
        bay = info.get("bay")
        stack = info.get("stack")
        containersMatrix = info.get("containersMatrix")
        headingTrucksNumber = info.get("headingTrucksNumber")
        queuingTrucksNumber = info.get("queuingTrucksNumber")
        headingContainers = info.get("headingContainers")
        queuingContainers = info.get("queuingContainers")
        taskNumber = info.get("taskNumber")
        relocationNumber = info.get("relocationNumber")
        # reward = info.get('reward')

        containerRelocationNumber=info.get("containerRelocationNumber")
        is_done = info.get('isDone')
        # print("containerRelocationNumber",containerRelocationNumber)

        if show_info:
            print("reset****")
            print("bay: ", type(bay), " : ", bay)
            print("stack: ", type(stack), " : ", stack)
            print("containerMatrix: ", type(containersMatrix), " : ", containersMatrix)
            print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
            print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
            print("headingContainers: ", type(headingContainers), " : ", headingContainers)
            print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
            print("relocationNumber/taskNumber: ", float(relocationNumber) / float(taskNumber))
            # print("reward: ", type(reward), " : ", reward)
            print("is_done: ", type(is_done), " : ", is_done)
        headingContainers = self.regulateStrings(headingContainers)
        queuingContainers = self.regulateStrings(queuingContainers)
        obs = Observation(bay, stack, containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers,
                          queuingContainers, relocationNumber)
        self.observation = obs
        s = self.getState(obs)
        return s

    def receive_end_info(self):
        self.client.send(str(-1).encode('GBK'))  # send episode final info request
        end_info = json.loads(str(self.client.recv(1024 * 10), encoding="GBK"))

    def regulateStrings(self, str):
        if str == None or len(str) == 0:
            str = "0000000000"
        elif len(str) == 5:
            str = str + "00000"
        else:
            str = str[0:10]
        return str

    def getState(self, obs):
        # bay   1
        # stack   1
        # headingTrucksNumber   1
        # queuingTrucksNumber   1
        # containersMatrix  150 =>6
        # headingContainers  10
        # queuingContainers  10
        # total 174

        # s=[obs.bay,obs.stack,obs.headingTrucksNumber,obs.queuingTrucksNumber]+[int(x) for x in obs.containersMatrix]+[int(x) for x in obs.headingContainers]+[int(x) for x in obs.queuingContainers]
        s1 = [obs.bay, obs.stack, obs.headingTrucksNumber, obs.queuingTrucksNumber]
        s2 = [int(x) for x in obs.containersMatrix]
        s2 = s2[obs.bay * 6:obs.bay * 6 + 6]
        # s2=s2[]
        s3 = [int(x) for x in obs.headingContainers]
        s4 = [int(x) for x in obs.queuingContainers]

        s = s1 + s2 + s3 + s4
        if show_info:
            print(len(s1), " ", len(s2), " ", len(s3), " ", len(s4))
            print("shape s: ", len(s), " ; ", s)
        s = np.array(s).astype(np.uint8)
        return s

    def checkAction(self,action:int):

        while (True):
            pileSize = int(self.observation.containersMatrix[self.observation.bay * 6 + action])


            if (action != self.observation.stack) and (pileSize <6):
                return action
            else:
                # action = np.random.randint(0, 6)

                action=(action+1)%6
                # print("change to ", action)
                # print("this should not prompt")

    def checkActionValid(self,action:int):
        pileSize = int(self.observation.containersMatrix[self.observation.bay * 6 + action])
        if (action != self.observation.stack) and (pileSize <6):
            return True
        else:
            return False



    def calculateReward(self,action,bay):
        r=0
        if action == self.observation.stack:
            # print("not move")
            r-=5

        # check the layer of new position
        # reward+=4-layer
        pileSize = int(self.observation.containersMatrix[self.observation.bay * 6 + action])
        r = 0.5-1.0/6.0*pileSize
        # print("pile$",r)

        # check whether the new position will be the reshuffled in near future
        # if so, reward-=10
        # print(self.observation.queuingContainers)
        # print("queue bay",self.observation.queuingContainers[1:3]," ",self.observation.queuingContainers[6:8],", bay ",bay)
        # print("queue stack",self.observation.queuingContainers[3]," ",self.observation.queuingContainers[8],", action ",action)
        #
        # print("heading bay",self.observation.headingContainers[1:3]," ",self.observation.headingContainers[6:8],", bay ",bay)
        # print("head stack",self.observation.headingContainers[3]," ",self.observation.headingContainers[8],", action ",action)
        if(int(self.observation.queuingContainers[1:3])==bay and self.observation.queuingContainers[3]==action) or (int(self.observation.queuingContainers[6:8])==bay and self.observation.queuingContainers[8]==action):
            r-=3
            # print("queue $")


        if (int(self.observation.headingContainers[1:3]) == bay and self.observation.headingContainers[3] == action) or (
                int(self.observation.headingContainers[1:3]) == bay and self.observation.headingContainers[8] == action):
            r -= 4
            # print("head $")

        # print("current $",r)
        return r

    def step(self, action: int):
        reward=0
        # print("============================")
        # check whether the the action is valid:
        # layer should less than 4
        # new position should different from th eold one
        if not self.checkActionValid(action):
            # print("!!!!!!!!!")
            if (self.env_type == "test"):
                # action = self.checkAction(action)
                action = (action + 1) % 6
            else:
                return self.getState(self.observation), -1000, False, {"validAction": False}

        # print("           1111111")
        self.client.send(str(action).encode('GBK'))
        # print("        22222222")
        info = json.loads(str(self.client.recv(1024), encoding="GBK"))
        # print("         333333")
        bay = info.get("bay")
        stack = info.get("stack")
        containersMatrix = info.get("containersMatrix")
        headingTrucksNumber = info.get("headingTrucksNumber")
        queuingTrucksNumber = info.get("queuingTrucksNumber")
        headingContainers = info.get("headingContainers")
        queuingContainers = info.get("queuingContainers")
        taskNumber = info.get("taskNumber")
        relocationNumber = info.get("relocationNumber")
        containerRelocationNumber=info.get("containerRelocationNumber")

        # reward = info.get('reward')
        is_done = info.get('isDone')
        # reward += self.calculateReward(action,bay)  #version 1


        if not is_done:
            headingContainers = self.regulateStrings(headingContainers)
            queuingContainers = self.regulateStrings(queuingContainers)
            if (show_info):
                print("state0: ")
                print("bay: ", type(bay), " : ", bay)
                print("stack: ", type(stack), " : ", stack)
                print("containerMatrix: ", type(containersMatrix), " : ", containersMatrix)
                print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
                print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
                print("headingContainers: ", type(headingContainers), " : ", headingContainers)
                print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
                print("relocationNumber: ", relocationNumber, " taskNumber: ", taskNumber)
                print(" relocationNumber/taskNumber: ", float(relocationNumber) / float(taskNumber))
                if (containerRelocationNumber > 0):
                    print("containerRelocationNumber", containerRelocationNumber)
                print("is_done: ", type(is_done), " : ", is_done)
            # reward -= ( float(relocationNumber) / float(taskNumber) + 0.07) * 100
            if(containerRelocationNumber>0):
                # reward-=1
                self.additionalRelocation+=1

            if(containerRelocationNumber>=0):   #version 1
                reward-=1
            # print("reward ",reward)
            obs = Observation(bay, stack, containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers,
                              queuingContainers, relocationNumber)

            self.observation = obs
            s_ = self.getState(obs)

            return s_, 0, is_done, {"validAction": True}
        else:
            # print("port ",self.port,": episode ",self.count," end, additionalRelocation: ",self.observation.relocationNumber)
            # self.count+=1
            # self.global_relocation_list.append(self.observation.relocationNumber)

            print("port ",self.port,": episode ",self.count," end, additionalRelocation: ",self.additionalRelocation)
            self.count+=1
            self.global_relocation_list.append(self.additionalRelocation)
            self.additionalRelocation=0
            cmean=getMean(self.global_relocation_list[-100:])

            if(self.count>=100 and cmean<self.bestMean):
                self.bestMean=cmean
                self.bestMeanIndex=self.count

            # print(self.global_relocation_list)
            if(self.env_type=="test"):
                self.relocation_list.append(self.observation.relocationNumber)
            return np.zeros(30).astype(np.uint8), -1, is_done, {"validAction": True}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    # def receive_end_info(self):
    #     self.client.send(str(-1).encode('GBK'))  # send episode final info request
    #     end_info = json.loads(str(self.client.recv(1024*10), encoding="GBK"))

# env = YardEnv(16, 10010, 'train')
# # If the environment don't follow the interface, an error will be thrown
#
# check_env(env, warn=True)
#
#
# # wrap it
# # env = make_vec_env(lambda: env, n_envs=1)
# # Train the agent
# model = A2C('MlpPolicy', env, verbose=1).learn(1000)
# # model.learn(1000)


# Test the trained agent
# obs = env.reset()
# n_steps = 20
# for step in range(n_steps):
#   action, _ = model.predict(obs, deterministic=True)
#   print("Step {}".format(step + 1))
#   print("Action: ", action)
#   obs, reward, done, info = env.step(action)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render(mode='console')
#   if done:
#     # Note that the VecEnv resets automatically
#     # when a done signal is encountered
#     print("Goal reached!", "reward=", reward)
#     break
