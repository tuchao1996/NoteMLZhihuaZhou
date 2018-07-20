<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 机器学习周志华 学习笔记

## 第一章 绪论

* 泛化能力
* 归纳偏好
* NFT(No Free Lunch Theorem)，所有算法对所有问题的期望是相同的，寓意需要结合具体案例选择适宜的算法求解。
* 发展历程

    1950，基于神经网络的“连接主义”(connectionism)，代表作为感知机(conceptron)。
    
    1960，基于逻辑表示的“符号主义”(symbolism)

    1980，机器学习成为一门独立的学科领域。

    1990，神经网络研究取得重大进展，BPNN出现，但参数设置缺乏理论指导。

    1995，统计学习(statistical learning)出场，代表作支持向量机(SVM)，一般化的核方法(kernel methods)。

    2000，深度学习浪潮涌来，设计语音、图像等应用广泛。火热原因：大数据，计算能力强，深度学习模型的参数众多，缺乏训练则容易过拟合；大数据时代，设备计算能力增强，数据储备和计算能力发展大。

    > 美国最尖端的技术由两大机构推进，NASA And DARPA。

## 第二章 模型评估与选择

### 2.1 经验误差与过拟合
    
    精度(accuracy)、训练误差(training error)、过拟合(overfitting)、欠拟合(underfitting)

### 2.2 评估方法

    * 留出法（hold-out），采用“分层采样”（stratified sampling）。
        > 一般可选择随机100次结果，返回结果的平均值。
        >
        > 常见方法：大约2/3-4/5用作训练。
    
    * 交叉验证法（cross validation）
        > 将数据集D划分为k个子集，子集之间不存在交集。然后，每次用k-1个子集并集作为训练集，余下的子集作为测试集。这样可获得k组测试结果，最终返回k个测试结果的均值。
        
        > 特例：留一法（leave-one-out）。即D中每个样本作为一个子集。
        
    * 自助法（bootstrapping）
    
        > 从m个样本中每次随机挑选一个后，加入D'中。该采样过程进行m次，得到和D相同大小的D'。由公式可知
        
$$ \lim_{m \to \infty}(1-\frac{1}{m}) \to \frac{1}{e} \approx 0.368 $$
        
        > D'中存在0.368概率的样本未被选中，因此可将D'作为训练集，D作为测试集。
        
        > 该方法适用于数据集较少的情况。
        
    * 调参与最终模型
        > 调参（parameter tuning），对参数可在一定范围内固定步长调整。
    
### 2.3 性能度量
    
    * 错误率与精度
    * 准确率（precision）与召回率（recall）
        > P-R 曲线
    
    * ROC（receiver operation characteristic）与AUC（area under-ROC curve） 

### 2.4 比较检验
    
    > 测试误差是否一定等于泛化误差？

### 2.5 偏差与方差（bias-variance decomposition）
    
    > 泛化误差可分解为偏差、方差与噪声之和。
    
    $$E(f;D) = bias^2(x)+var(x)+\epsilon^2$$
    
    > 训练程度小时，偏差大，方差小，泛化误差大；
    >
    > 反之相反。
    
## 第三章 线性模型

### 3.1 基本形式

基本的数学表达形式如下所示：

$$f(x)=w^Tx+b$$

### 3.2 线性回归（linear regression）

令均方误差（mean square loss，MSE）最小，是回归任务的常用性能度量。

几何意义对应“欧式距离最小”（Euclidean distance），基于MSE进行模型求解的方法为“最小二乘法”（least square method）。

闭式解（closed-form）如下：

$$
w = \frac{\sum_{i=1}^{m}y_{i}(x_{i}-\overline{x})}{\sum_{i=1}^{m}x_{i}^2-\frac{1}{m}(\sum_{i=1}^{m}x_{i})^2}

b=\frac{1}{m}\sum_{i=1}^{m}(y_{i}-wx_{i})
$$

线性模型可变形，如“对数线性回归”（log-linear regression）。

$$lny=w^Tx+b$$

### 3.3 对数几率回归（logistic regression）

Sigmoid函数

$$y=\frac{1}{1+e^{-z}}$$

该函数对应为“对数几率回归”，是一种分类方法。

### 3.4 线性判别分析