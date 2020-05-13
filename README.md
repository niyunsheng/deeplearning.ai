# deeplearning.ai

5门课程一共16周的课程，完成课后作业，并且整理笔记。

这是一个相对完整的课程体系，比自己随便找资料效率更高。**coursera的课程费用需要每月$49，但是真正的成本是你的时间和精力！而且付费还能给你一定的监督，没必要省这点钱。**

作业和视频至少一样重要！

[网易云课程视频](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)

参考前辈笔记：
* 主要参考[黄海广（包含全部的字幕）](http://www.ai-start.com/dl2017/)
* [黄海广github](https://github.com/fengdu78/deeplearning_ai_books)
* 主要参考[大树先生笔记](https://zhuanlan.zhihu.com/p/35333489)
* [!KyonHuang](http://kyonhuang.top/Andrew-Ng-Deep-Learning-notes/#/)
* [红色石头](https://zhuanlan.zhihu.com/p/36453627)
* [毛帅](http://imshuai.com/tag/deeplearning-ai-notes/)


参考作业答案
* 主要参考[JudasDie/deeplearning.ai](https://github.com/JudasDie/deeplearning.ai)
* 主要参考[大树先生作业](https://blog.csdn.net/Koala_Tree/article/category/7186915)
* [enggen/Deep-Learning-Coursera](https://github.com/enggen/Deep-Learning-Coursera)
* [bighuang624](https://github.com/bighuang624/Andrew-Ng-Deep-Learning-notes/tree/master/assignments)

推荐学习过程：视频+黄飞海笔记=》自己的笔记，高质量完成作业。

本人学习过程：作业为主，视频和笔记为辅。

# C1W1 深度学习引言

神经网络来表征函数关系，通过大规模数据训练找到最优表示方程。


事实上如今最可靠的方法来在神经网络上获得更好的性能，往往就是要么训练一个更大的神经网络，要么投入更多的数据。

# C1W2：神经网络基础

首先用逻辑回归 logistic regression 来表达前向传播和反向传播(forward/backward propagation) 的想法。

logistic regression是一个二分类算法（可以理解为输出是或者不是），所以叫做逻辑回归。




作业：Logistic Regression with a Neural Network mindset

$x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$


$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$

Forward Propagation:
- You get X
- You compute $A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})$
- You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$

backward:

$$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
$$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

本次作业用逻辑回归二分类损失函数的SVM网络，每次forward和backward都采用全部的数据，选用一定的学习率，用梯度下降法进行参数更新。

# W3：浅层神经网络

作业：Planar data classification with one hidden layer

即有一个隐层的神经网络。

数据并不是线性分开的，所以用sklearn的线性分类器来分类效果很差，而采用神经网络可以发现分类的特征。

$x^{(i)}$:
$$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1] (i)}\tag{1}$$
$$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
$$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2] (i)}\tag{3}$$
$$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
$$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$

Given the predictions on all the examples, you can also compute the cost $J$ as follows:
$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$



# W4：深层神经网络

作业：


# 无监督学习

K-means

迭代算法，做两件事，
* 簇分配：检查所有的数据点，看数据点距离哪个聚类中心最近。
* 移动聚类中心：找到所有同类点，然后算出他们的中心作为新的聚类中心。

输入：k、一群无标签数据集

Kmeans的优化目标函数：失真代价函数，就是所有的点距离其聚类中心的距离的平方和的均值。

可以在数学上证明，这个簇分配的步骤就是最小化代价函数J。
移动聚类中心所做的就是选择u值来最小化J。

随即初始化：避免局部最优解。

随即初始化可能会局部最优解：可能将一个簇分为两个，把两个簇看做一个。

如何选择聚类数量？
很多都是手动。

不同的人观察，聚类数也是模棱两可的。

肘部原则：改变K，计算代价函数J。画出曲线，然后找拐点。

用Kmeans来市场分割等，根据目的给出一个评估标准，能更好的用于后续事件。比如我只把t恤分三种，那么K就等于3。

# 课程2
# W1：深度学习的使用层面
# W2：优化算法
# W3：超参数调试、Batch正则化和程序框架

# 课程3

# W1：机器学习策略（上）
# W2：机器学习策略（下）

# 课程4

# W1：卷积神经网络
# W2：深度卷积网络：实例探究
# W3：目标检测




# W4：特殊应用：人脸识别和神经风格转换

# 课程5

# W1：循环序列模型
# W2：自然语言处理与词嵌入
# W3：序列模型和注意力机制
