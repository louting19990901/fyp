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
        truckToQCDistance = info.get('truckToQCDistance')
        qcRemainTaskAmount = info.get('qcRemainTaskAmount')
        qcTruckQueueLength = info.get('qcTruckQueueLength')
        currentWorkingTruckAmount = info.get('currentWorkingTruckAmount')
        shipAmount = info.get('shipAmount')
        headingToQCAmount = info.get('headingToQCAmount')
        qcTypes = info.get('qcTypes')

        #state = self.generate_feature_vector([truckToQCDistance, qcRemainTaskAmount, qcTruckQueueLength, currentWorkingTruckAmount])
        state = self.generate_feature_vector_and_action_space([truckToQCDistance,
                                                                           qcRemainTaskAmount,
                                                                           qcTruckQueueLength,
                                                                           currentWorkingTruckAmount,
                                                                           headingToQCAmount,
                                                                           qcTypes])

        reward = info.get('reward')
        is_done = info.get('isDone')

        return state

    # def generate_feature_vector(self, feature_list):
    #     feature_vector = np.array(feature_list).T
    #     return feature_vector

    def step(self, action:int):

        self.client.send(str(action).encode('GBK'))
        #print(self.env_type, 'client sent action to port', self.port)

        info = json.loads(str(self.client.recv(1024), encoding="GBK"))

        # -----------state-----------
        truckToQCDistance = info.get('truckToQCDistance')
        # print("truckToQCDistance: ", truckToQCDistance)

        qcRemainTaskAmount = info.get('qcRemainTaskAmount')
        # print("qcRemainTaskAmount: ", qcRemainTaskAmount)

        qcTruckQueueLength = info.get('qcTruckQueueLength')
        currentWorkingTruckAmount = info.get('currentWorkingTruckAmount')
        #operationTimes = info.get('taskOperationTimes')
        shipAmount = info.get('shipAmount')
        headingToQCAmount = info.get('headingToQCAmount')
        qcTypes = info.get('qcTypes')
        #state = self.generate_feature_vector([truckToQCDistance, qcRemainTaskAmount, qcTruckQueueLength, currentWorkingTruckAmount])
        state = self.generate_feature_vector_and_action_space([truckToQCDistance,
                                                                           qcRemainTaskAmount,
                                                                           qcTruckQueueLength,
                                                                           currentWorkingTruckAmount,
                                                                           headingToQCAmount,
                                                                           qcTypes])

        reward = info.get('reward')
        is_done = info.get('isDone')
        return state, reward, is_done

    def generate_feature_vector_and_action_space(self, state):
        full_length_feature_vector = np.array(state).T
        return full_length_feature_vector

    def receive_end_info(self):
        self.client.send(str(-1).encode('GBK'))  # send episode final info request
        end_info = json.loads(str(self.client.recv(1024*10), encoding="GBK"))
        # ------------- end info --------------
        wait_time = dict()
        total_length = 0
        for qc_index in range(self.qc_num):
            times = end_info.get(str(qc_index)).split(' ')
            wait_time[str(qc_index)] = times
            total_length += len(times)

        return wait_time

    def render(self, mode='human'):
        pass

    def close(self):
        #os.chdir("..")
        pass

    def test_port(self):
        self.executor.test()


def play(env):
    s = env.reset()
    total_r = 0
    while(True):

        # ------ select action from model here ------
        act = random.randint(0, 15)

        s_, r, done = env.step(act)
        total_r += r
        s = s_
        if done:
            env.receive_end_info()
            break
    return total_r


if __name__ == "__main__":
    env = Env(16, 10005, 'train')
    score = play(env)
    print('score ', score)