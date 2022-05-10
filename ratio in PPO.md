# Ratio in PPO

之前看到 baselines 的 ppo2 一直有个地方没懂，就是这里对 ratio 的计算：[https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#**L78**](https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py#L78)。注意这里是 $\pi_{old} - \pi$

```python
# Calculate ratio (pi current policy / pi old policy)
ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
```

今天对 ppo2 进行了 debug 才发现原来 `OLDNEGLOGPAC` 和 `neglogpac` 的值均是正的。但一般来说，`log_prob` 的值应该是负的，也就是说 baselines 的实现中加了一个符号把数值变为正值，所以反过来。

## Debug ppo2 in baselines

首先，搭建一个 baselines 的虚拟环境，然后执行

```
cd baselines/baselines
python run.py --alg=ppo2 --env=HalfCheetah-v2 --num_timesteps=1e6
```