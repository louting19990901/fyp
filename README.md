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

- [ ] 

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

- [ ] 在python端读取数据

- [ ] 实现python函数RLGetAction

- [ ] 存储(state,reward)

- [ ] 修改receive_end_info()

- [ ] 修改最后一次的getAction()  in checkExit()

- [ ] 循环多个episode

- [ ] 删掉无关信息（优先度低）

- [ ] 创建state类in python

- [ ] debug：python读取数据ing

  





# RL

- state： 当前堆场的使用情况，箱子的信息，任务列表
- action：90% 选择Q值最高的，10%随机选择，
  - 不能选择已经满的堆叠
  - action为一个bay位
- reward 为单个箱子的翻箱次数
- debug  python 环境没有运行中断









更改status

















