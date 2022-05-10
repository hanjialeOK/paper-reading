# A Natural Policy Gradient

[https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)

TRPO 的理论基础

## 海森矩阵

维基百科上讲的害行

二元函数 $f(x_1,x_2)$ 在 $(m,n)$ 点的泰勒展开式为

$$f(x_1,x_2)=f(m,n)+f'_{x_1}(m,n)\Delta x_1+f'_{x_2}(m,n)\Delta x_2 + \frac{1}{2} \left[ f''_{x_1}(m,n)\Delta x_1^2 + 2f''_{x_1x_2}(m,n)\Delta x_1 \Delta x_2 + f''_{x_2}(m,n)\Delta x_2^2 \right] + \cdots$$

其中，$\Delta x_1=x_1-m$，$\Delta x_2=x_2-m$，把上述展开式写成矩阵形式有

$$f(x)=f(x_0)+\nabla f(x_0)^T \Delta x + \frac{1}{2}\Delta x^T G(x_0) \Delta x + \cdots$$

其中，$\Delta x =\begin{bmatrix} \Delta x_1 \\[8pt] \Delta x_2 \end{bmatrix}$，$\nabla f(x_0)=\begin{bmatrix} \frac{\partial f}{\partial x_1} \\[8pt] \frac{\partial f}{\partial x_2} \end{bmatrix}$ 是函数 $f(x_1,x_2)$ 在 $(m,n)$ 处的梯度。矩阵

$$G(x_0)=\nabla^2 f(x_0)=\begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\[8pt] \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} \end{bmatrix}_{x_0}$$

即函数 $f(x_1,x_2)$ 在 $x_0(m,n)$ 处的 $2\times 2$ 海森矩阵。

现在将二元函数的泰勒展开式推广到多元函数，那么

$$\nabla f(x_0)=\begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}_{x_0}^T$$

为函数 $f$ 在 $x_0$ 点处的梯度

$$G(x_0)=\nabla^2 f(x_0)=\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[8pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[8pt]
\vdots & \vdots & \ddots & \vdots \\[8pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n \partial x_n} \\[8pt]
\end{bmatrix}_{x_0}$$

## 牛顿法

维基百科上讲的害行，下面的 gif 也是来自维基百科。

牛顿法一般用来求解 $f(x)=0$ 的根。首先选择一个接近 $f(x)$ 零点的 $x_0$，计算相应的 $f(x_0)$ 和斜率 $f'(x_0)$，然后计算通过点 $(x_0,f(x_0))$ 并且斜率为 $f'(x_0)$ 的直线与 $x$ 轴的交点横坐标，即求解：

$$0=(x-x_0)f'(x_0)+f(x_0)$$

将得到的横坐标记为 $x_1$，通常 $x_1$ 会比 $x_0$ 更接近 $f(x)=0$ 的解。然后利用 $x_1$ 开启下一轮迭代。公式可以化简如下：

$$x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}$$

当然，牛顿法能够收敛也是有条件的，这里就不细说了。

![Newton](NewtonIteration_Ani.gif)

下面，转而使用牛顿法去计算极值点，也就是求解 $f'(x)=0$。对于简单的一维情况，将 $f(x)$ 泰勒展开

$$f(x)=f(x_0)+f'(x_0)(x-x_0)+\frac{1}{2}f''(x_0)(x-x_0)^2$$

两边同时求导可得

$$f'(x)=f'(x_0)+f''(x_0)(x-x_0)$$

令 $f'(x)=0$ 可得 $x=x_0-\frac{f'(x_0)}{f''(x_0)}$

把这个点记为 $x_1$，然后重复上述过程有

$$x_{n+1}=x_n-\frac{f'(x_n)}{f''(x_n)}$$

好了，现在把情况推广到多元函数：$x$ 和 $x_0$ 均为列向量，将 $f(x)$ 泰勒展开

$$f(x)=f(x_0)+\nabla f(x_0)^T(x-x_0)+\frac{1}{2}(x-x_0)^T \nabla^2 f(x_0)(x-x_0)$$

两边同时求导可得

$$\nabla f(x)=\nabla f(x_0)+\nabla^2 f(x_0)(x-x_0)$$

令 $\nabla f(x)=0$ 得 $x=x_0-\left[\nabla^2 f(x_0)\right]^{-1} \nabla f(x_0)$。

不难看出，$\nabla^2 f(x_0)$ 就是海森矩阵啊，那么就用 $H$ 表示，用 $g$ 表示梯度 $\nabla f(x_0)$。从而

$$x_{n+1}=x_n-H_n^{-1}g_n$$

$-H^{-1}g$ 被称为**牛顿方向**。通过解决 $f'(x)=0$ 的问题，那么牛顿法就可以找到函数的极值，也就是用于梯度下降！

## Fisher Information Matrix

费希尔信息，用来衡量随机变量 $X$ 携带的关于未知参数 $\theta$ 的信息量，其中 $X$ 的概率分布依赖于参数 $\theta$。这是维基百科的定义。

假设有概率分布 $p(x;\theta)$，真实参数 $\theta$ 未知，但是 $x$ 可以观测到。

$$\begin{aligned}
E \left[ \frac{\partial}{\partial \theta} \log f(x;\theta) \Bigg| \theta \right] &= \int_R \frac{\nabla f(x;\theta)}{f(x;\theta)} f(x;\theta)dx \\
&= \nabla \int_R f(x;\theta)dx \\
&= \nabla 1 = 0
\end{aligned}$$

记 $g(x;\theta)=\nabla \log f(x;\theta)=\frac{\nabla f(x;\theta)}{f(x;\theta)}$，其实这就是 score function，当参数 $\theta$ 是列向量时，它也是一个列向量。然后定义费希尔信息

$$I(\theta)=E \left[ g(x;\theta)g(x;\theta)^T \right]$$

不难发现，费希尔信息其实就是 score function 的方差

$$\begin{aligned}
Var[g(x;\theta)]&=E[g(x;\theta)g(x;\theta)^T]-E[g(x;\theta)]E[g(x;\theta)]^T \\
&=E[g(x;\theta)g(x;\theta)^T] \\
&= I(\theta)
\end{aligned}$$

然后最重要的来了

$$I(x;\theta)=-E[\nabla^2 \log f(x;\theta)]$$

证明如下

$$\begin{aligned}
\nabla^2 \log f(x;\theta) &= \nabla \left[ \frac{\nabla f(x;\theta)}{f(x;\theta)} \right] \\
&= \frac{\nabla^2f(x;\theta)f(x;\theta)-\nabla f(x;\theta) [\nabla f(x;\theta)]^T}{[f(x;\theta)]^2} \\
&= \frac{\nabla^2f(x;\theta)}{f(x;\theta)}-\left[ \frac{\nabla f(x;\theta)}{f(x;\theta)} \right] \left[ \frac{\nabla f(x;\theta)}{f(x;\theta)} \right]^T \\
&= \frac{\nabla^2f(x;\theta)}{f(x;\theta)}-\nabla \log f(x;\theta) [\nabla \log f(x;\theta)]^T
\end{aligned}$$

两边同时求期望

$$\begin{aligned}
E \left[ \nabla^2 \log f(x;\theta) \right] &= E \left[ \frac{\nabla^2f(x;\theta)}{f(x;\theta)} \right] - E \left[ g(x;\theta)g(x;\theta)^T \right] \\
&= \int_R \frac{\nabla^2f(x;\theta)}{f(x;\theta)} f(x;\theta)dx - I(\theta) \\
&= \nabla^2 \int_Rf(x;\theta)dx - I(\theta) \\
&= -I(\theta)
\end{aligned}$$

## KL 散度

首先证明 KL 散度不小于零

$$\begin{aligned}
KL(f||g)&=\int f(x) \ln \frac{f(x)}{g(x)}dx \\
&=-\int f(x) \ln \frac{g(x)}{f(x)}dx \\
&\geq -\int f(x) \left[\frac{g(x)}{f(x)}-1 \right]dx \qquad \ln x \leq x-1\\
&= - \int (g(x) - f(x)) dx \\
&= 0
\end{aligned}$$

现在，我们考虑梯度下降前分布 $f(x;\theta)$ 和 梯度下降后分布 $f(x;\theta+\Delta \theta)$ 的变化。我们希望参数更新前后，$f(x;\theta)$ 和 $f(x;\theta+\Delta \theta)$ 的变化不要太大。我们可以用 KL 散度来衡量分布 $f(x;\theta)$ 和 $f(x;\theta+\Delta\theta)$ 的差异。

$$\begin{aligned}
KL(f(x;\theta)||f(x;\theta+\Delta\theta))&=\int f(x;\theta) \log \left( \frac{f(x;\theta)}{f(x;\theta+\Delta\theta)} \right)dx \\
&=\int f(x;\theta) \log f(x;\theta) dx - \int f(x;\theta) \log f(x;\theta+\Delta\theta)dx
\end{aligned}$$

泰勒展开

$$f(\theta+\Delta\theta) \approx f(\theta)+\nabla_\theta f(\theta)^T\Delta\theta+\frac{1}{2}\Delta\theta^T \nabla_\theta^2f(\theta) \Delta\theta$$

同理，

$$\log f(x;\theta+\Delta\theta) \approx \log f(x;\theta) + \nabla_\theta\log f(x;\theta)^T \Delta\theta + \frac{1}{2}\Delta\theta^T \nabla_\theta^2\log f(x;\theta) \Delta\theta$$

于是，我们可以继续推

$$\begin{aligned}
KL(f(x;\theta)||f(x;\theta+\Delta\theta)) &= \int f(x;\theta) \log f(x;\theta) dx - \int f(x;\theta) \log f(x;\theta+\Delta\theta)dx \\
&\approx-\int f(x;\theta) \nabla_\theta\log f(x;\theta)^T \Delta\theta dx - \frac{1}{2} \Delta\theta^T \left( \int f(x;\theta) \nabla_\theta^2\log f(x;\theta) dx \right) \Delta\theta \\
&=- \frac{1}{2} \Delta\theta^T \left( \int f(x;\theta) \nabla_\theta^2\log f(x;\theta) dx \right) \Delta\theta \\
&=- \frac{1}{2} \Delta\theta^T E \left[ \nabla^2 \log f(x;\theta) \right] \Delta\theta \\
&=\frac{1}{2} \Delta\theta^T I(\theta) \Delta\theta
\end{aligned}$$

KL 顺序变化会有什么影响？
