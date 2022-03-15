import socket
import jpype
import os
import json
import random
import time
import numpy as np
# import torch


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
        s=Observation(bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers)

        #state = self.generate_feature_vector_and_action_space([containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers, queuingContainers])
        reward = info.get('reward')
        is_done = info.get('isDone')

        print("reset****")
        #print("state0: ",state)
        print("bay: ", type(bay), " : ", bay)
        print("stack: ", type(stack), " : ", stack)
        print("containerMatrix: ",type(containersMatrix)," : ",containersMatrix)
        print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
        print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
        print("headingContainers: ", type(headingContainers), " : ", headingContainers)
        print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
        print("reward: ", type(reward), " : ", reward)
        print("is_done: ", type(is_done), " : ", is_done)


        return s

    # def generate_feature_vector(self, feature_list):
    #     feature_vector = np.array(feature_list).T
    #     return feature_vector

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
        s_=Observation(bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers)


        #state = self.generate_feature_vector_and_action_space([containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers])

        reward = info.get('reward')
        is_done = info.get('isDone')
        print("state0: ")
        print("bay: ", type(bay), " : ", bay)
        print("stack: ", type(stack), " : ", stack)
        print("containerMatrix: ", type(containersMatrix), " : ", containersMatrix)
        print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
        print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
        print("headingContainers: ", type(headingContainers), " : ", headingContainers)
        print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
        print("reward: ", type(reward), " : ", reward)
        print("is_done: ", type(is_done), " : ", is_done)
        return s_, reward, is_done

    def generate_feature_vector_and_action_space(self, state):
        full_length_feature_vector = np.array(state).T
        return full_length_feature_vector

    def receive_end_info(self):
        self.client.send(str(-1).encode('GBK'))  # send episode final info request

        end_info = json.loads(str(self.client.recv(1024*10), encoding="GBK"))
        # ------------- end info --------------
        # wait_time = dict()
        # total_length = 0
        # for qc_index in range(self.qc_num):
        #     times = end_info.get(str(qc_index)).split(' ')
        #     wait_time[str(qc_index)] = times
        #     total_length += len(times)

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

    # while(True):
    #     print("obs: ",observation.bay," ",act," shape")
    #     for i in range(25):
    #         for j in range(6):
    #             print(observation.containersMatrix[i*6+j],end="")
    #         print()

        pileSize = int(observation.containersMatrix[observation.bay*6+act])
        print(pileSize)
        if(act != observation.stack) and (pileSize < 7):
            return act
        else:
            act=random.randint(0, 5)



def play(env):
    s = env.reset()
    total_r = 0
    while(True):

        # ------ select action from model here ------
        #act = random.randint(0, 6)
        act=RLGetAction(s)
        s_, r, done = env.step(act)
        total_r += r
        s = s_
        if done:
            env.receive_end_info()
            break
    return total_r

class Observation:
    bay = -1
    stack=-1
    containersMatrix=""
    headingTrucksNumber=-1
    queuingTrucksNumber=-1
    headingContainers=""
    queuingContianers=""


    def __init__(self,bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers):
        self.stack=stack
        self.bay=bay
        self.containersMatrix=containersMatrix
        self.headingTrucksNumber=headingTrucksNumber
        self.queuingTrucksNumber=queuingTrucksNumber
        self.headingContainers=headingContainers
        self.queuingContianers=queuingContainers


if __name__ == "__main__":
    env = Env(16, 10005, 'train')
    score = play(env)
    print('score ', score)