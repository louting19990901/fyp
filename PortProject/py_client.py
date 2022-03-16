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

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000

N_ACTIONS = 6
N_STATES = 174


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

    def getState(self,obs):
        #bay   1
        #stack   1
        #headingTrucksNumber   1
        #queuingTrucksNumber   1
        #containersMatrix  150
        #headingContainers  10
        #queuingContainers  10
        #total 174
        s=[obs.bay,obs.stack,obs.headingTrucksNumber,obs.queuingTrucksNumber]+[int(x) for x in obs.containersMatrix]+[int(x) for x in obs.headingContainers]+[int(x) for x in obs.queuingContainers]
        s=np.array(s)
        print("shape s: ",s.shape)
        return s


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
        obs=Observation(bay,stack,containersMatrix,headingTrucksNumber,queuingTrucksNumber,headingContainers,queuingContainers)

        s=self.getState(obs)


        return s,obs

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

        # reward = info.get('reward')
        is_done = info.get('isDone')


        if not is_done:
        #     print("state0: ")
        #     print("bay: ", type(bay), " : ", bay)
        #     print("stack: ", type(stack), " : ", stack)
        #     print("containerMatrix: ", type(containersMatrix), " : ", containersMatrix)
        #     print("headingTrucksNumber: ", type(headingTrucksNumber), " : ", headingTrucksNumber)
        #     print("queuingTrucksNumber: ", type(queuingTrucksNumber), " : ", queuingTrucksNumber)
            # print("headingContainers: ", type(headingContainers), " : ", headingContainers)
            # print("queuingContainers: ", type(queuingContainers), " : ", queuingContainers)
            headingContainers = self.regulateStrings(headingContainers);
            queuingContainers = self.regulateStrings(queuingContainers)
            print("relocationNumber: ",relocationNumber," taskNumber: ",taskNumber)
            print(" relocationNumber/taskNumber: ",float(relocationNumber)/float(taskNumber))
            obs = Observation(bay, stack, containersMatrix, headingTrucksNumber, queuingTrucksNumber, headingContainers,queuingContainers)

        #     print("reward: ", type(reward), " : ", reward)
        #     print("is_done: ", type(is_done), " : ", is_done)
        #reward=float(relocationNumber)/float(taskNumber)
        else:
            return -1,-1,is_done,-1
        reward=0
        s_=self.getState(obs)
        print("end")
        return s_, reward, is_done,obs

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
    act = np.random.randint(0, 6)

    while(True):
        pileSize = int(observation.containersMatrix[observation.bay*6+act])

        print(pileSize)
        if(act != observation.stack) and (pileSize < 7):
            return act
        else:
            act=np.random.randint(0, 6)





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











class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def del_tensor_ele(self,arr,index):
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
        return torch.cat((arr1,arr2),dim=0)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()

            #check whether the action(stack) is available, otherwise, choose the second largest q value
            pile=
            if(action==s[1] or x[4+x[0]+action]>=7):
                index_max=torch.max(actions_value,1)[0]
                action=torch.max(self.del_tensor_ele(actions_value,index_max),1)[1].data.numpy()

        else:  # random
            action = np.random.randint(0, N_ACTIONS)

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":

    env = Env(16, 10005, 'train')
    for episode in range(1000):
        print(" =====episode ",episode,"=====")
        s,obs = env.reset()

        while (True):

            # ------ select action from model here ------

            act = RLGetAction(obs)
            s_, r, done,obs = env.step(act)
            # total_r += r
            s = s_
            if done:
                env.receive_end_info()
                break
