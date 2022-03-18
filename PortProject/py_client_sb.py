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

class Observation:   #total 174 bit
    bay = -1                    #1 bit
    stack=-1                    #1 bit
    containersMatrix=""         #6*25 =150 bit
    headingTrucksNumber=-1      #1 bit
    queuingTrucksNumber=-1      #1 bit
    headingContainers=""    #10 bit
    queuingContainers=""    #10 bit



    def __init__(self,bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers):
        self.stack=stack
        self.bay=bay
        self.containersMatrix=containersMatrix
        self.headingTrucksNumber=headingTrucksNumber
        self.queuingTrucksNumber=queuingTrucksNumber
        self.headingContainers=headingContainers
        self.queuingContainers=queuingContainers

class YardEnv(gym.Env):
    
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_actions, port, env_type):
        super(YardEnv, self).__init__()
        self.client = None
        self.port = port
        self.is_connected = False
        self.qc_num = n_actions
        self.env_type = env_type
        self.root_path = os.getcwd()
        self.executor = self.start_java_end()

        self.action_space = spaces.Discrete(6)
        # Example for using image as input (channel-first; channel-last also works):
        arr=[24,5,10,10]
        for i in range(150):
            arr.append(0)
        for i in range(20):
            arr.append(4)
        
        self.observation_space = spaces.Box(low=0, high=25,
                                            shape=(174,), dtype=np.uint8)

        # start server and client
    def start_java_end(self):
        os.chdir("JavaProject/bin")
        jvmPath = jpype.getDefaultJVMPath()
        jar_path = '-Djava.class.path={}'.format(
            self.get_jars(self.root_path))
        if not jpype.isJVMStarted():
            jpype.startJVM(jvmPath, jar_path)
        print(os.getcwd())
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
        os.chdir("..")
        executor = javaClass(self.port, self.env_type)  # get an instance of java object

        #javaInstance = javaClass.getInstance('demo')  # invoke class static function directly

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
    
        # start server and simulation
        self.executor.startServer()

        # connect to server
        if not self.is_connected:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect(('127.0.0.1', self.port))
            self.is_connected = True
            #print('client connected')

        info = json.loads(str(self.client.recv(1024), encoding="GBK"))
        print("8888")

        # -----------state-----------
        bay = info.get("bay")
        stack = info.get("stack")
        containersMatrix = info.get("containersMatrix")
        headingTrucksNumber = info.get("headingTrucksNumber")
        queuingTrucksNumber = info.get("queuingTrucksNumber")
        headingContainers = info.get("headingContainers")
        queuingContainers = info.get("queuingContainers")
        taskNumber=info.get("taskNumber")
        relocationNumber=info.get("relocationNumber")
        reward = info.get('reward')
        is_done = info.get('isDone')
        # print("reset****")
        # print("bay: ", type(bay), " : ", bay)
        # print("stack: ", type(stack), " : ", stack)
        # print("containerMatrix: ",type(containersMatrix)," : ",containersMatrix)
        # print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
        # print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
        # print("headingContainers: ", type(headingContainers), " : ", headingContainers)
        # print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)

        print("relocationNumber/taskNumber: ",float(relocationNumber)/float(taskNumber))
        # print("reward: ", type(reward), " : ", reward)
        # print("is_done: ", type(is_done), " : ", is_done)
        headingContainers = self.regulateStrings(headingContainers)
        queuingContainers = self.regulateStrings(queuingContainers)
        obs=Observation(bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers)
        s=self.getState(obs)
        # print("state: ",s.shape)
        return s

    def receive_end_info(self):
        self.client.send(str(-1).encode('GBK'))  # send episode final info request
        end_info = json.loads(str(self.client.recv(1024*10), encoding="GBK"))

    def regulateStrings(self,str):
        if len(str)==0:
            str="0000000000"
        elif len(str)==5:
            str=str+"00000"
        else:
            str=str[0:10]
        return str
    
    def getState(self,obs):
        #bay   1
        #stack   1
        #headingTrucksNumber   1
        #queuingTrucksNumber   1
        #containersMatrix  150
        #headingContainers  10
        #queuingContainers  10
        #total 174

        # s=[obs.bay,obs.stack,obs.headingTrucksNumber,obs.queuingTrucksNumber]+[int(x) for x in obs.containersMatrix]+[int(x) for x in obs.headingContainers]+[int(x) for x in obs.queuingContainers]
        s1=[obs.bay,obs.stack,obs.headingTrucksNumber,obs.queuingTrucksNumber]
        s2=[int(x) for x in obs.containersMatrix]
        s3=[int(x) for x in obs.headingContainers]
        s4=[int(x) for x in obs.queuingContainers]
        print(len(s1)," ",len(s2)," ",len(s3)," ",len(s4))
        s=s1+s2+s3+s4

        print("shape s: ",len(s)," ; ",s)
        s=np.array(s).astype(np.uint8)
        return s
    
    def step(self, action:int):
        print("5555")
        print("action is ",action)
        self.client.send(str(action).encode('GBK'))
        print("4444")
        info = json.loads(str(self.client.recv(1024), encoding="GBK"))
        print("6666")
        bay=info.get("bay")
        stack=info.get("stack")
        containersMatrix=info.get("containersMatrix")
        headingTrucksNumber=info.get("headingTrucksNumber")
        queuingTrucksNumber=info.get("queuingTrucksNumber")
        headingContainers=info.get("headingContainers")
        queuingContainers=info.get("queuingContainers")
        taskNumber = info.get("taskNumber")
        relocationNumber = info.get("relocationNumber")

        # reward = info.get('reward')
        is_done = info.get('isDone')
        if not is_done:
        #     print("state0: ")
            print("bay: ", type(bay), " : ", bay)
            print("stack: ", type(stack), " : ", stack)
        #     print("containerMatrix: ", type(containersMatrix), " : ", containersMatrix)
        #     print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
        #     print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
            # print("headingContainers: ", type(headingContainers), " : ", headingContainers)
            # print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
            headingContainers = self.regulateStrings(headingContainers)
            queuingContainers = self.regulateStrings(queuingContainers)
            print("relocationNumber: ",relocationNumber," taskNumber: ",taskNumber)
            print(" relocationNumber/taskNumber: ",float(relocationNumber)/float(taskNumber))
        #     print("reward: ", type(reward), " : ", reward)
        #     print("is_done: ", type(is_done), " : ", is_done)
            reward=float(relocationNumber)/float(taskNumber)
            obs = Observation(bay, stack, containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers,queuingContainers)
            s_=self.getState(obs)
            return s_, reward, is_done,{}
        else:            
            return np.zeros(174).astype(np.uint8),-1,is_done,{}

    def render(self, mode='human'):
        pass 
    
    def close (self):
        pass
    
    def receive_end_info(self):
        
        self.client.send(str(-1).encode('GBK'))  # send episode final info request
        end_info = json.loads(str(self.client.recv(1024*10), encoding="GBK"))
        
env = YardEnv(16, 10006, 'train')
# If the environment don't follow the interface, an error will be thrown
print("12345")
check_env(env, warn=True)
print("hello")