# HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION [ICLR 2016]

[https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

## Words

reinforcement learning **literature** \
强化学习**文献**

in **a variety of** different ways. \
使用**大量的**方法

since the effect of an action **is confounded with** the effects of past and future actions. \
因为一个动作的效果与过去和未来动作的效果**相混淆**

**variance** \
**方差**

But while high variance **necessitates** using more samples, **bias** is more **pernicious**——even with an unlimited number of samples, bias can cause the algorithm to fail to **converge**, or to converge to a poor solution that is not even a local optimum \
但是，虽然高方差**需要**使用更多样本，但**偏差**更加**有害**——即使样本数量不受限制，偏差也会导致算法无法**收敛**，或者收敛到甚至不是局部最优的糟糕解。

We propose a family of policy gradient estimators that **significantly** reduce variance while maintaining a **tolerable** level of bias. \
我们提出了一系列策略梯度估计器，它们可以**显著**减少方差，同时保持**可容忍的**偏差水平。

**The contributions of this paper are summarized as follows:** \
本文的贡献总结如下：

**scheme** \
**方案**

**While** the formula has been pro- posed in prior work, our analysis is **novel** and enables GAE to be applied with a more general set of algorithms, including the batch trust-region algorithm we use for our experiments. \
**虽然**该公式已在先前的工作中提出，但我们的分析是**新颖的**，并且使 GAE 能够应用于更通用的算法集，包括我们用于实验的批量信任区域算法

By combining (1) and (2) **above**, we obtain an algorithm that **empirically** is effective at learning neural network policies for challenging control tasks. The results **extend the state of the art** in using reinforcement learning for high-dimensional continuous control. \
通过结合**上面的** (1) 和 (2)，我们获得了一种算法，该算法在**经验上**有效地学习用于具有挑战性的控制任务的神经网络策略。 结果**扩展了使用强化学习进行高维连续控制的最新技术**

## Tips

### Variance and bias

When using a parameterized stochastic policy, it is possible to obtain an **unbiased** estimate of the gradient of the expected total returns \
当使用参数化随机策略时，可以获得预期总收益梯度的无偏估计

Actor-critic methods, use a value function rather than the empirical returns, ob- taining an estimator with lower variance at the cost of introducing bias \
actor-critic 方法使用价值函数而不是经验回报，以引入偏差为代价获得具有较低方差的估计量

But while high variance necessitates using more samples, bias is more pernicious—even with an unlimited number of samples, bias can cause the algorithm to fail to con- verge, or to converge to a poor solution that is not even a local optimum \
但是，虽然高方差需要使用更多样本，但偏差更加有害——即使样本数量不受限制，偏差也会导致算法无法收敛，或者收敛到甚至不是局部最优的糟糕解

### Contribution

- Propose $\lambda$
- Propose the use of a **trust region optimization** method for the value **function**

### Why actor-critic have a lower variance ?

The choice $\Psi_t = A_\pi(s_t, a_t)$ yields almost the **lowest possible variance**, though in practice, the advantage function is not known and must be estimated. \
选择 $\Psi_t = A_\pi(s_t, a_t)$ 产生的方差几乎是最低的，尽管在实践中，优势函数是未知的，必须估计

This statement can be intuitively justified by the following interpretation of the policy gradient: that a step in the policy gradient direction should increase the probability of better-than-average actions and decrease the probability of worse-than-average actions. The advantage function, by it’s definition $A_\pi(s, a) = Q_\pi(s, a) −V_\pi(s)$, measures whether or not the action is better or worse than the policy’s default behavior. Hence, we should choose $\Psi_t$ to be the advantage function $A_\pi(s_t, a_t)$, so that the gradient term $\Psi_t \nabla_\theta \log \pi_\theta (a_t | s_t)$ points in the direction of increased $\pi_\theta (a_t | s_t)$ if and only if $A_\pi(s_t, a_t) > 0$. See Greensmith et al. (2004) for a more rigorous analysis of the variance of policy gradient estimators and **the effect of using a baseline**. \
通过以下对策略梯度的解释可以直观地证明这一说法是合理的：在策略梯度方向上的一步应该增加优于平均水平的动作的概率并降低低于平均水平的动作的概率。根据定义，优势函数 $A_\pi(s, a) = Q_\pi(s, a) −V_\pi(s)$ 衡量该行为是比策略的默认行为好还是坏。 因此，我们应该选择 $\Psi_t$ 作为优势函数 $A_\pi(s_t, a_t)$，使得梯度项 $\Psi_t \nabla_\theta \log \pi_\theta (a_t | s_t)$ 指向 $\pi_\theta (a_t | s_t)$ 增加的方向当且仅当 $A_\pi(s_t, a_t) > 0$。参见 Greensmith 等人（2004）对策略梯度估计的方差和使用基线的影响进行了更严格的分析

We will introduce a parameter $\gamma$ that allows us to reduce variance by downweighting rewards corresponding to delayed effects, at the cost of introducing bias. This parameter corresponds to the discount factor used in discounted formulations of MDPs, but we treat it as a variance reduction parameter in an undiscounted problem \
我们将引入一个参数 $\gamma$，它允许我们通过降低与延迟效应相对应的奖励权重来减少方差，但代价是引入偏差。 该参数对应于 MDP 折扣公式中使用的折扣因子，但我们将其视为未折扣问题中的方差缩减参数

### $\lambda$

$$
\begin{aligned}
&\hat{A}_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots + \cdots + (\gamma \lambda)^{T-t+1} \delta_{T-1} \\
&\text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\end{aligned}
$$
$$
\begin{aligned}
&\lambda = 0: \quad \hat{A}_t = \delta_t &&= r_t + \gamma V(s_{t+1}) - V(s_t) \\
&\lambda = 1: \quad \hat{A}_t = \sum_{l = 0}^{\infin} \gamma^l \delta_{t+l} &&= \sum_{l = 0}^{\infin} \gamma^l r_{t+l} - V(s_t)
\end{aligned}
$$

$\lambda = 0$, induces bias, but it typically has much lower variance. \
$\lambda = 1$, regardless of the accuracy of $V$, but it has high variance due to the sum of terms. \
The generalized advantage estimator for $0 < \gamma < 1$ makes a compromise between bias and variance, controlled by parameter $\gamma$.

### Value function estimation

we used a trust region method to optimize the value function in each iteration of a batch optimization procedure.

$$
\text{minimize}_{\phi} = \sum_{n=1}^N ||V_\phi (s_n) - \hat{V}_n || ^2 \\
\text{subject to } \frac{1}{N} \sum_{n=1}^N \frac{||V_\phi (s_n) - V_{\phi_{old}} (s_n) || ^2}{2 \sigma^2} \leq \epsilon
$$

### Why policy update using old value parameters

Note that the policy update $\theta_i \to \theta_{i+1}$ is performed using the value function $V_{old}$ for advantage estimation, not $V_{new}$. Additional bias would have been introduced if we updated the value function first. To see this, consider the extreme case where we overfit the value function, and the Bellman residual $r_t + \gamma V(s_{t+1}) −V(s_t)$ becomes zero at all timesteps—the policy gradient estimate would be zero.
