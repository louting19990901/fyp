### 

- [x] pathmind =》 break
- [x] Bonsai =》no
- [x] 导出anylogic为.jar文件
- [x] 使用socket 将java和python连接通讯     

### PortProject1.1





- [x] 查看getAction使用方法

- [x] 使用随机函数负责卡车调度

- [ ] 使用getAction控制堆场操作

- [ ] 将任务生成文件嵌入java文件，自动执行，改为每次随机初始任务（优先级低）

- [ ] 设计强化学习伪代码

- [ ] 在python文件实现Q-learning强化学习逻辑

- [ ] 实验

- [ ] 增大堆场中已有箱子的数量，提高翻箱率

- [x] 查看yard中bay25，stack6，layer4的取值范围

- [x] 更改status成分（当前堆场箱子堆叠情况，目标箱子的位置，与当前yard有关的任务列表） =》定义Observation

- [x] 将observation代替status传到java

- [x] 将observation转码为json

- [x] 将json转到python

- [x] reward修改：系统总翻箱数*-1

- [x] 找到管理yard的数据格式

- [x] 找到表示箱子位置的数据格式

- [x] 找到yard表示已有箱子的方式

- [x] count relocationTime

- [x] 处理双箱问题（Q：一个卡车上的双箱是从两个不同yard取，然后放置到两个不同yard吗)

- [x] 找到并提取task中箱子位置的参数，

- [x] 搞懂sameSource意味着什么：allocateSpaceInYard

- [x] Q： 翻箱只在同一bay位进行，是为了让大小箱不搞混吗

- [x] Q：source和location的区别

- [x] Q:为什么在java的task里面source的类型是int，但是在anylogic的task里面source的类型是location

- [x] Q：怎么对比source和yard

- [x] Q：为什么location能转化为yard？   （yard)location

- [x] observation    对比yard和source（考虑双箱），将相应的箱子的目标bay，stack，layer 存在observation中

- [x] 找到为什么只有双箱需要翻箱

- [x] 在python端读取数据

- [x] 实现python函数RLGetAction

- [ ] 存储(state,reward)

- [x] 修改receive_end_info()

- [x] 修改最后一次的getAction()  in checkExit()

- [x] 循环多个episode

- [x] 删掉无关信息（优先度低）

- [x] 创建state类in python

- [x] debug：python读取数据ing

- [x] 在observation中加入taskNumber和relocationTime

- [x] 在communicator.java中发送taskNumber，relocationNumber

- [x] 在python中接收taskNumber，relocationNumber并计算比值

- [x] 在send_end_info.java中加入observation

- [ ] 合并pythonclient端代码与模范DQN代码

- [x] 将observation转化为numpy

- [x] chooseAction  修改

- [x] dbeug:reward null:其实是因为没有处理好idDone=true的情况下传回来的参数

- [x] 修改store_transition

  

### stable-baseline3 

- [x] 通过check_env()
- [x] 找出action_sample的替代方法
  - [x] try:当出现不合理的action时，给一个超大的惩罚，令isDone=false
  - [x] try: 将之前返回的150个参数变为5个：原本一个bay应该是6个stack，但是只要记录下原本stack的位置，sample的时候在5个action中选择一个；坏处：但是还要解决layer>4的问题，结构并没有改变： 好处：传输的信息变少（加快model速度），无关参数减少（提高训练效率）
  - [x] debug:解决n_proxs问题+路径问题（第二次不需要再定位文件位置）：失败
  - [x] sub_vec_make:需要尝试注册自定义env： 失败
- [x] evaluate helper
- [ ] 考虑通过fix random锁定环境 
- [ ] 每个几代evaluate一次并画图
- [ ] 并行 dummy
- [ ] 删掉anylogic中的flowControl fail to contain truck语句







# RL

- state： 当前堆场的使用情况，箱子的信息，任务列表
- action：90% 选择Q值最高的，10%随机选择，
  - 不能选择已经满的堆叠
  - action为一个bay位
- reward 为单个箱子的翻箱次数
- debug  python 环境没有运行中断



- sparse reward 问题如何解决
- 与其类似的围棋问题
- 但是与机械手臂栓螺丝问题不同，该问题一开始的时候由于参数是随机初始化的，几乎不可能拴好，因此基本无法出现positive的reward
- 解决方法：定义额外reward，也叫reward shaping
- 在yard的情境下，由于在一个episode中任务数量是不变的，因此一下参数可以作为reward shaping的额外reward
  - reshuffleNumber/taskNumber
- 给机器加上好奇心：碰到新事物可以增加reward
- 对比 action mask 和自创的返回重新选择
- 







更改status







### wrapper

- 设置max episode
- normalize the action space\





### DQN 算法

- [ ] A2C
- [ ] DQN
- [ ] PPO
- [ ] ji



## Training and testing environment





# reward

- [ ] 根据relocationNumber来计算reward
  - [x] reward=-(翻箱数/任务数-baseline)
  - [x] reward=log(翻箱数/任务数-baseline)
  - [x] 在anylogic中计算当前设置最少的翻箱数，reward=预计翻箱数-实际翻箱数
- [ ] 当container放置的位置不合适时，reward=-1000
  - [x] layer>4
  - [x] 翻箱前后位置未改变
- [ ] 根据layer来计算reward
  - [ ] 翻箱后当前pile的layer（1<=layer<=6)越小越好 ，reward=6-layer
- [ ] 翻箱后的container尽量不影响需要被取走的container
  - [ ] 如果翻箱后的container在要被取走的container上方，reward=-10
- [ ] 





# REFLECTION

- 和supervised learning的区别
- sparsed reward情况下reward的设置
- reward delay
- exploration
  - add noise
  - use big discount factor









# PPO

- the actor to train has to know its difference from the actor to interact
- https://youtu.be/OAKAZhFmYol



### DQN

https://youtu.be/o_g9JUMw1Occ

https://youtu.be/2-zGCx4iv_k









1. state encoding
2. preliminaries
