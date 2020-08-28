# PytorchTrainAcceleration
一个基于Pytorch的项目示例，主要：
- 尝试各种针对轻量化模型训练的加速方案
- 提供了一个供参考的处理流程
- 实现了一些可复用的基础模块

## 1. 功能架构

总体入口为`./src/train_model.py`，需要根据自己的环境进行配置`config_file`。

### 1.1 Config接口

用`json`文件来设置所有配置参数，实现配置参数的层级管理。

在[./config](./config/)文件夹下，可以根据训练不同配置和阶段保存不同的`config`文件。

### 1.2 Data相关接口

在[src/datasets](./src/datasets/)文件夹下，实现功能包括：

基于第三方API[NVIDIA-DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html)的支持，Dataset、Transform和Dataloader的功能，最终实例化一个dataloader的迭代器，可以输出`torch.tensor`格式的数据，与通常的training过程接轨。

### 1.3 Model接口

#### 1.3.1 网络结构定义

在[src/models](./src/models/)文件夹下， 自定义网络结构。

可将自己实现的网络结构，按照`torch.nn.Module`的标准方式实现。  

#### 1.3.2 模型初始化

##### 1.3.2.1 模型参数加载

支持如下几种模型参数加载方法：

- 载入checkpoint：
  - 如果指定了checkpoint，或者搜索到checkpoint，优先加载，
- 载入预训练模型：
  - 然后看是否有预训练的model_file，
- 随机初始化
  - 如果都没有，则没有特定操作，采用模型本身的随机初始化方法。

##### 1.3.2.2 冻结部分参数

根据分支或节点名称，冻结参数，不参与反向传播。

##### 1.3.2.3 Data parallel

用`nn.parallel.DistributedDataParallel`代替较为旧的`nn.DataParallel`模块。  
按照[官方推荐](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)，新的并行化方法显卡的负载更均衡，速度也更快。

> 因为nn.DataParallel基于 Parameter-server 算法，只是将数据分成了N份，在推理时实现并行，推理的结果会合并到0号GPU上进行loss计算，因此，0号GPU的负载会明显大于其他GPU，还会有不小的通信开销。  
> 如果训练的配置不合理，有可能出现多卡反而比单卡慢的情况。

### 1.4 Loss接口

在[src/losses](./src/losses)文件夹下，自定义Loss的实现方式。

### 1.5 Optimizer及Lr接口

#### 1.5.1 Optimizer设定

可以在`config`文件中配置不同的 optimizer。

#### 1.5.2 Learning Rate设置

目前采用在每一个训练的 epoch 后动态调整 learning rate 的策略。

### 1.6 其他辅助训练功能

集中在[src/train](./src/train/)文件夹下，包括如下功能：

#### 1.6.1 saver接口

主要用来处理训练过程中checkpoint的相关功能，包括：

- 检查和记录当前checkpoint状态
- 保存checkpoint
- 载入checkpoint

#### 1.6.2 logger接口

将训练过程中的loss等参数记录到tensorboard的log文件中。  

#### 1.6.3 evaluator接口

根据模型推理输出和GroundTruth计算性能指标。  
实现了一个用于分类项目的[`Evaluator`](./src/train/evaluator.py)作为示例。

#### 1.6.4 utils

其他通用处理模块，如：

- 文件IO：如读取并解码json文件
- logging：针对多进程的打印输出模块

### 1.7 其他资源

#### 1.7.1  训练过程记录

在[./train](./train/)文件夹下，保存：

- 训练的checkpoint文件
- 训练的log文件
- 部分数据集生成的文件列表

#### 1.7.2 模型文件

在[./model_file](./model_file/)文件夹下，保存：

- 下载或自己生成的预训练模型参数文件
- 训练输出的模型参数文件

## 2. 运行使用

进行数据集加载测试：
```
scripts/test_datasets.sh
```

执行训练：
```
scripts/train_with_ddp.sh
```

针对每一个不同的项目：

- `datasets`，`models`是肯定要添加相应实现的模块；
- `losses`，`logger`，`evaluator`是可以做到一定程度共用的模块；
- 其他是可以做到较大范围共用的模块。