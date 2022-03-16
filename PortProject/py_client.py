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


class Env():

    metadata = {'render.modes': ['human']}

    def __init__(self, n_actions, port, env_type):
        super(Env, self).__init__()
        self.client = None
        self.port = port
        self.is_connected = False
        self.qc_num = n_actions
        self.env_type = env_type
        self.root_path = os.getcwd()
        self.executor = self.start_java_end()

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
        # if ('F:\PortProject' == os.getcwd()):
        #     os.chdir('JavaProject')

        # start server and simulation
        self.executor.startServer()

        # connect to server
        if not self.is_connected:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect(('127.0.0.1', self.port))
            self.is_connected = True
            #print('client connected')

        info = json.loads(str(self.client.recv(1024), encoding="GBK"))

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
        s=Observation(bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers,relocationNumber,taskNumber)

        #state = self.generate_feature_vector_and_action_space([containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers, queuingContainers])
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
        if not is_done:
            print("relocationNumber/taskNumber: ",float(relocationNumber)/float(taskNumber))
        # print("reward: ", type(reward), " : ", reward)
        # print("is_done: ", type(is_done), " : ", is_done)



        return s

    # def generate_feature_vector(self, feature_list):
    #     feature_vector = np.array(feature_list).T
    #     return feature_vector

    def regulateStrings(self,str):
        if len(str)==0:
            str="0000000000"
        elif len(str)==5:
            str=str+"00000"
        else:
            str=str[0:10]
        return str


    def step(self, action:int):

        self.client.send(str(action).encode('GBK'))
        info = json.loads(str(self.client.recv(1024), encoding="GBK"))

        bay=info.get("bay")
        stack=info.get("stack")
        containersMatrix=info.get("containersMatrix")
        headingTrucksNumber=info.get("headingTrucksNumber")
        queuingTrucksNumber=info.get("queuingTrucksNumber")
        headingContainers=info.get("headingContainers")
        queuingContainers=info.get("queuingContainers")
        taskNumber = info.get("taskNumber")
        relocationNumber = info.get("relocationNumber")
        s_ = Observation(bay, stack, containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers,queuingContainers, relocationNumber, taskNumber)

        reward = info.get('reward')
        is_done = info.get('isDone')

        if not is_done:
        #     print("state0: ")
        #     print("bay: ", type(bay), " : ", bay)
        #     print("stack: ", type(stack), " : ", stack)
        #     print("containerMatrix: ", type(containersMatrix), " : ", containersMatrix)
        #     print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
        #     print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
            headingContainers=self.regulateStrings(headingContainers);
            queuingContainers=self.regulateStrings(queuingContainers)

            print("headingContainers: ", type(headingContainers), " : ", headingContainers)
            print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
            print("relocationNumber: ",relocationNumber," taskNumber: ",taskNumber," relocationNumber/taskNumber: ",float(relocationNumber)/float(taskNumber))
        #     print("reward: ", type(reward), " : ", reward)
        #     print("is_done: ", type(is_done), " : ", is_done)


        return s_, reward, is_done

    def generate_feature_vector_and_action_space(self, state):
        full_length_feature_vector = np.array(state).T
        return full_length_feature_vector

    def receive_end_info(self):
        self.client.send(str(-1).encode('GBK'))  # send episode final info request
        end_info = json.loads(str(self.client.recv(1024*10), encoding="GBK"))



    def render(self, mode='human'):
        pass

    def close(self):
        #os.chdir("..")
        pass

    def test_port(self):
        self.executor.test()

def RLGetAction(observation):
    #print("RL get action...")
    act = random.randint(0, 5)

    while(True):
        pileSize = int(observation.containersMatrix[observation.bay*6+act])
        print(pileSize)
        if(act != observation.stack) and (pileSize < 7):
            return act
        else:
            act=random.randint(0, 5)





class Observation:
    bay = -1                    #1 bit
    stack=-1                    #1 bit
    containersMatrix=""         #本来是6*25 =》1*25
    headingTrucksNumber=-1      #1 bit
    queuingTrucksNumber=-1      #1 bit
    headingContainers=""    #10 bit
    queuingContianers=""    #10 bit
    relocationNumber=-1
    taskNumber=-1


    def __init__(self,bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers,relocationNumber,taskNumber):
        self.stack=stack
        self.bay=bay
        self.containersMatrix=containersMatrix
        self.headingTrucksNumber=headingTrucksNumber
        self.queuingTrucksNumber=queuingTrucksNumber
        self.headingContainers=headingContainers
        self.queuingContianers=queuingContainers
        self.relocationNumber=relocationNumber
        self.taskNumber=taskNumber










class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 6)
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        return self.fc(inputs)

if __name__ == "__main__":

    env = Env(16, 10005, 'train')
    for episode in range(1000):
        print(" =====episode ",episode,"=====")
        s = env.reset()

        while (True):

            # ------ select action from model here ------

            act = RLGetAction(s)
            s_, r, done = env.step(act)
            # total_r += r
            s = s_
            if done:
                env.receive_end_info()
                break
