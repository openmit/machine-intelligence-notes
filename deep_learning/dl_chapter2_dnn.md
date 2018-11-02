## 深度神经网络（DNN）


### 基于梯度下降的反向传播算法


| 符号 | 含义 |
| :--: | :-- |
| $n^{[0]}$ | 表示NN输入层的神经元个数，为了符号统一，规定输入层为第0层 |
| $n^{[l]}$ | 第$l$层神经元个数 |
| $a^{[l]}$ | 表示NN第$l$层输入信息，一条样本的维度为$n^{[l]}$
| $W^{[l]}$ | 表示第$l$层与$l-1$层之间的连接矩阵，维度：($n^{[l]}$, $n^{[l-1]}$)，即：$W^{[1]} \in R^{n^{[1]} \times n^{[0]}}$ | 
| $b^{[l]}$ | 表示第$l$层的偏置项参数，维度：($n^{[l]}$, $1$) | 
| $z^{[l]}$ | 表示第$l$层的线性表达$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$, 维度：($n^{[l]}$, $1$) | 


$$
a^{[l]} = \sigma(z^{[l]})  这里\sigma为激活函数表达式
$$

前向计算

$$
z^{[1]} = W^{[1]} \cdot a^{[0]} + b^{[1]},  \\\
a^{[1]} = \sigma(z^{[1]}), \\\
z^{[2]} = W^{[2]} \cdot a^{[1]} + b^{[2]},  \\\
a^{[2]} = \sigma(z^{[2]}), \\\
z^{[3]} = W^{[3]} \cdot a^{[2]} + b^{[3]},  \\\
a^{[3]} = \sigma(z^{[3]}), \\\
\hat{y} = sigmoid(a^{[3]})
$$

反向计算

$$
\mathcal{L}(a^{[3]};W^{[1]},\; \cdots,\;b^{[3]}) = - y \cdot \log(\hat{y}) - (1 - y) \cdot \log(1-\hat{y})  \\\
\mathbf{d}a^{[3]} = \frac{\partial{\mathcal{L}}}{\partial a^{[3]}} = \frac{\partial{\mathcal{L}}}{\partial{\sigma(z^{[3]})}}  = (\hat{y} - y)  \\\
\mathbf{d}z^{[3]} = \mathbf{d}a^{[3]} \cdot \frac{{\partial{a^{[3]}}} }{\partial z^{[3]}}, \quad  注：\frac{\partial{a^{[3]}}} {\partial z^{[3]}}是一个关于z^{[3]}的梯度公式，也是一个关于输入z^{[3]}和输出z^{[3]}的表达式  \\\
\ma
$$




参考：http://www.cnblogs.com/southtonorth/archive/2018/08/21/9512559.html