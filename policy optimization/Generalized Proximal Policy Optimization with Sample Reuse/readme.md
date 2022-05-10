# Generalized Proximal Policy Optimization with Sample Reuse

[https://arxiv.org/abs/2111.00072](https://arxiv.org/abs/2111.00072)
评分：6 6 7 7

从理论层面证明了 PPO 使用旧数据复用的限制条件，从而是性能大幅提升。

## 指出问题

Online RL （原文用词是 on-policy）如 PPO 是有理论保证的，但是蒙特卡洛方法带来的高方差要求每次更新时需要大量的数据，但是 PPO 每次只能提供 2048 个步长的数据。\
Offline RL (原文用词是 off-policy）通过一个很大的 memory pool 可以解决数据的采样问题，但是会造成数据分布漂移（distribution shiff），因为采样的数据可能是很久以前的策略产生的。这对于 offline RL 影响不大，因为这些算法如DQN, DDPG,往往是单步的状态动作价值函数算法（Sutton 中文版254页），理论上不合理，但实际上成功使用。但是这种分布漂移对 online 算法是致命的，因为它破坏了 PPO 算法的理论保证！

那么能不能从理论上来结合具有理论保证的 online 算法和采样高效的 offline 算法呢？这篇文章提出 GePPO 解决了这个问题。这篇工作的三点贡献：

1. 如上
2. 找到了 clip 机制与公式第二项的联系，也就是在理论上证明了PPO-clip算法能work的原因，并且用 clip 机制来解决如上问题
3. 提出了一种自适应学习率方法，其实这种方法 PPO 那篇文章里已经提到了，这里只是变换一下if条件语句

## Preliminaries

值得一提的是，在进行公式推导前有一个假设：

> We model the agent’s decisions as a stationary policy π

也就是说是在平稳的前提下，那么“平稳”这个条件到底影响的是哪一部分？

策略改进下限公式：

$\pi$ represents any future policy \
$\pi_k$ represents a current policy \
$C^{\pi, \pi_k}=\max_{s \in \mathcal{S}} |\mathbb{E}_{a \sim \pi(\cdot | s)}[A^{\pi_k}(s,a)]|$ \
$TV(\pi, \pi_k)(s)$ represents the total variation distance between the distributions $\pi(\cdot | s)$ and $\pi_k(\cdot |s)$

$$
J(\pi) - J(\pi_k) \geq \underbrace{\frac{1}{1-\gamma} \mathop{\mathbb{E}}_{(s, a) \sim d^{\pi_k}} \left[ \frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s,a) \right]}_{\text{surrogate objective}} - \underbrace{\frac{2 \gamma C^{\pi,\pi_k}}{(1-\gamma)^2} \mathbb{E}_{s \sim d^{\pi_k}}\left[TV(\pi, \pi_k)(s)\right]}_{\text{penalty term}} $$

`surrogate objective` 这个短语在 PPO 那篇文章里被提及很多次，这是因为 PPO 优化的就是这一部分：

$$
L_k^{PPO}=\mathbb{E}_{(s, a) \sim d^{\pi_k}} \left[\min \left(\frac{\pi(a|s)}{\pi_k(a|s)}A^{\pi_k}(s,a), \text{clip}\left( \frac{\pi(a|s)}{\pi_k(a|s)}, 1-\epsilon, 1+\epsilon \right) A^{\pi_k}(s,a) \right)\right]
$$

文章对公式的第二项 `penalty term` 进行了推导，变相证明了 PPO 中 clip 机制。

$$
\mathbb{E}_{s \sim d^{\pi_k}}\left[TV(\pi, \pi_k)(s)\right] = \frac{1}{2} \mathbb{E}_{(s, a) \sim d^{\pi_k}} \left[ \left|\frac{\pi(a|s)}{\pi_k(a|s)} - 1 \right| \right]
$$

Generalized policy improvement lower bound：

$\pi$ represents any future policy \
$\pi_k$ represents a current policy \
$C^{\pi, \pi_k}=\max_{s \in \mathcal{S}} |\mathbb{E}_{a \sim \pi(\cdot | s)}[A^{\pi_k}(s,a)]|$ \
$TV(\pi, \pi_k)(s)$ represents the total variation distance between the distributions $\pi(\cdot | s)$ and $\pi_k(\cdot |s)$

$$
J(\pi) - J(\pi_k) \geq \underbrace{\frac{1}{1-\gamma} \mathbb{E}_{i \sim \nu} \Bigg[ \mathop{\mathbb{E}}_{(s, a) \sim d^{\pi_{k-i}}} \left[ \frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s,a) \right] \Bigg]}_{\text{surrogate objective}} - \underbrace{\frac{2 \gamma C^{\pi,\pi_k}}{(1-\gamma)^2} \mathbb{E}_{i \sim \nu} \Bigg[ \mathbb{E}_{s \sim d^{\pi_{k-i}}}\left[TV(\pi, \pi_{k-i})(s)\right] \Bigg]}_{\text{penalty term}}
$$

对上式中的第二项，应用三角不等式：

$$\mathbb{E}_{i \sim \nu} \Bigg[ \mathbb{E}_{s \sim d^{\pi_{k-i}}}\left[TV(\pi, \pi_{k-i})(s)\right] \Bigg] \leq \mathbb{E}_{i \sim \nu} \Bigg[ \mathbb{E}_{s \sim d^{\pi_{k-i}}}\left[TV(\pi, \pi_{k})(s)\right] \Bigg] + \sum_{j=1}^{M-1}\sum_{i=j}^{M-1} \nu_i \mathbb{E}_{s \sim d^{\pi_{k-i}}} [TV(\pi_{k-j+1}, \pi_{k-j})(s)]$$
