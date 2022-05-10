# Policy Gradient Methods for Reinforcement Learning with Function Approximation

[https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

提出 actor-critic 架构，建立策略梯度理论，并且证明这种方法是可以收敛到局部最优的。

## 指出问题

以往的强化方法中，所有的函数估计全部用来估计价值函数，然后通过估计的价值使用贪心算法选择动作。这种方法虽然好但是有一定限制：

- 这种方法是面向(be orientied toward)寻找确定性（determinstic）策略的，然而最优的策略往往是随机的，也就是说，对不同的动作往往有不同的选择概率。
- 对于估计的价值，如果有任意很小的改变，都可能会影响动作的选择。

## 策略梯度理论

$S_t \in \mathcal{S}$ \
$a_t \in \mathcal{A}$ \
$\mathcal{R}_s^a = E \{ r_{t+1}\ |\ s_t=s, a_t=a \}, \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$ \
$\mathcal{P}_{ss'}^a = Pr \{ s_{t+1}=s'\ |\ s_t=s, a_t=a \}, \quad \forall s, s' \in \mathcal{S}, a \in \mathcal{A}$ \
$\pi(s, a, \theta) = Pr \{ a_t=a\ |\ s_t=a, \theta \}, \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$

这里有一个假设，就是 $\pi$ 对 $\theta$ 导数是存在的，也即 $\frac{\partial \pi(s, a)}{\partial \theta}$ 存在。

文章里提出了两种智能体的目标（objective）.第一种是所有状态下的**平均奖励**

$$\rho(\pi)=\lim_{n \to \infty} \frac{1}{n} E \{ r_1 + r_2 + \cdots + r_n\ |\ \pi \} = \sum_s d^\pi(s) \sum_a \mathcal{R}_s^a$$

$$Q^\pi(s,a)=\sum_{t=1}^\infty E \{ r_t - \rho(\pi)\ |\ s_0=s, a_0=a, \pi \}, \quad \forall s \in \mathcal{S}, a \in \mathcal{A}.$$

这里是第二个假设，在策略 $\pi$ 下，假设状态服从平稳分布( stationary distribution) : $d^\pi(s)=\lim_{t \to \infty} Pr \{ s_t=s\ |\ s_0, \pi \}$

第二种是指定**开始状态** $s_0$，通过带折扣系数的蒙特卡洛计算长期收益

$$\rho(\pi)=E \left\{ \sum_{t=1}^\infty \gamma^{t-1}r_t\ \Bigg|\ s_0, \pi \right\}$$
$$Q^\pi(s, a)=E \left\{ \sum_{k=1}^{\infty} \gamma^{k-1}r_{t+k}\ \Bigg|\ s_t=s, a_t=a, \pi \right\}$$

这种情况下，我们定义

$d^\pi(s)=\sum_{t=0}^\infty \gamma^t Pr\{ s_{t}=s\ |\ s_0, \pi \}$

这里注意 $d^\pi(s)$ 是后来定义的。无论是对平均奖励还是指定开始状态的情况，均有策略梯度定理成立

$$\nabla_\theta \rho(\pi) = \sum_s d^\pi(s) \sum_a \nabla_\theta \pi(a|s) Q^\pi(s,a)$$

值得一提的是，$d^\pi(s)$ 对于 $\theta$ 是没有梯度的，也就是说，策略改变对于状态分布没有影响。因此，如果 $s$ 是服从策略 $\pi$ 的采样，那么 $\sum_a \nabla_\theta \pi(a|s) Q^\pi(s,a)$ 是对 $\nabla_\theta \rho(\pi)$ 的无偏估计。当然了，$Q^\pi(s,a)$ 本身我们肯定是不知道的，需要估计，至于是不是无偏那就另说了。实际上，我们一般会用采样的得到的 $R_t=\sum_{k=1}^\infty \gamma^{k-1} r_k$ 来作为 $Q^\pi(s_t,a_t)$ 的估计。这里文章里没有细说，只提到了修正项 $\frac{1}{\pi(s_t,a_t)}$，sutton 第二版中文版 323 页有推导

$$\begin{aligned}
\nabla J(\theta) &= \mathbb{E}_\pi \left[ \sum_a \pi(a|S_t, \theta) q_\pi(S_t,a) \frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)} \right] \\
&= \mathbb{E}_\pi \left[ q_\pi (S_t, A_t) \frac{\nabla \pi(A_t|S_t, \theta)}{\pi(A_t|S_t, \theta)} \right] \qquad 用采样\ A_t \sim \pi\ 替换\ a
\end{aligned}$$

或者写成这样更好理解一点

$$\begin{aligned}
\nabla_\theta \rho(\pi) &= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi(a|s) Q^\pi(s,a) \\
&= \mathbb{E}_{s_t \sim d^\pi} \left[ \sum_a \nabla_\theta \pi(a|s_t) Q^\pi(s_t,a) \right] \qquad 用采样\ s_t \sim d^\pi\ 替换\ s \\
&= \mathbb{E}_{s_t \sim d^\pi} \left[ \sum_a \pi(a|s_t) Q_\pi(s_t,a) \frac{\nabla_\theta \pi(a|s_t)}{\pi(a|s_t)} \right]\\
&= \mathbb{E}_{s_t \sim d^\pi} \Bigg[ \mathbb{E}_{a_t \sim \pi} \bigg[ Q_\pi(s_t,a_t) \frac{\nabla_\theta \pi(a_t|s_t)}{\pi(a_t|s_t)} \bigg] \Bigg] \qquad 用采样\ a_t \sim \pi\ 替换\ a
\end{aligned}$$

## 附录证明

首先证明指定开始状态的情况

$$\begin{aligned}
\nabla_\theta V^\pi(s)&=\nabla_\theta \left[ \sum_a \pi(a|s) Q^\pi(s,a) \right] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) + \pi(a|s)\nabla_\theta Q^\pi(s,a) \Big] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \sum_a \left[ \pi(a|s) \nabla_\theta \Big[ \mathcal{R}_s^a + \gamma \sum_{s'} Pr \{ s'|s,a \} V^\pi(s') \Big] \right] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \sum_a \left[ \pi(a|s) \nabla_\theta \Big[\gamma \sum_{s'} Pr \{ s'|s,a \} V^\pi(s') \Big] \right] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \gamma \sum_a \left[ \pi(a|s) \sum_{s'} Pr \{ s'|s,a \} \nabla_\theta V^\pi(s') \right] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \gamma \sum_a \left[ \pi(a|s) \sum_{s'} Pr \{ s'|s,a \} \nabla_\theta \Big[ \sum_{a'} \pi(a'|s') Q^\pi(s',a') \Big] \right] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \gamma \sum_a \left[ \pi(a|s) \sum_{s'} Pr \{ s'|s,a \} \sum_{a'} \Big[ \nabla_\theta \pi(a'|s') Q^\pi(s',a') + \pi(a'|s')\nabla_\theta Q^\pi(s',a') \Big] \right] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \gamma \sum_a  \pi(a|s) \sum_{s'} Pr \{ s'|s,a \} \sum_{a'} \Big[ \nabla_\theta \pi(a'|s') Q^\pi(s',a') \Big] + \gamma \sum_a  \pi(a|s) \sum_{s'} Pr \{ s'|s,a \} \sum_{a'} \Big[ \pi(a'|s')\nabla_\theta Q^\pi(s',a') \Big] \\
&=\sum_a \Big[ \nabla_\theta \pi(a|s) Q^\pi(s,a) \Big] + \gamma \sum_{s'} Pr \{ s \to s', 1, \pi \} \sum_{a'} \Big[ \nabla_\theta \pi(a'|s') Q^\pi(s',a') \Big] + \gamma^2 \sum_{s''} Pr \{ s \to s'', 2, \pi \} \sum_{a''} \Big[ \nabla_\theta \pi(a''|s'') Q^\pi(s'',a'') \Big] + \cdots \\
&=\sum_x \sum_{k=0}^\infty \gamma^k Pr \{ s \to x, k, \pi \} \sum_a \nabla_\theta \pi(a|x) Q^\pi(x,a) \\
\nabla_\theta V^\pi(s_0) &=\sum_x \sum_{k=0}^\infty \gamma^k Pr \{ s_0 \to x, k, \pi \} \sum_a \nabla_\theta \pi(a|x) Q^\pi(x,a) \\
&= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi(a|s) Q^\pi(s,a)
\end{aligned}$$
