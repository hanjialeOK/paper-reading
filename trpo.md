# TRPO

Policy gradient is the steepest ascend in parameter space with Euclidean metric:

$$
d^* = \nabla_\theta J(\theta) = \lim_{\epsilon \to 0} \frac{1}{\epsilon} \argmax J(\theta + d), \ s.t. \ ||d|| \leq  \epsilon
$$

Steepest ascend in distribution space (policy output) with KL-divergence as constraint:

$$
d^* = \argmax J(\theta + d), \ s.t. \ KL(\pi_\theta || \pi_{\theta + d}) = c
$$

$$
\begin{aligned}
d^* &= \argmax_d J(\theta + d) - \lambda (KL(\pi_\theta || \pi_{\theta + d}) - c) \\
&\approx \argmax_d J(\theta) + \nabla_\theta J(\theta)^T d - \frac{1}{2} \lambda d^T F d + \lambda c
\end{aligned}
$$

natural policy gradient:

$$
d = \frac{1}{\lambda} F^{-1} \nabla_\theta J(\theta)
$$

$$
\theta_{t + 1} = \theta_t + \alpha F^{-1} \nabla_\theta J(\theta), \ where \ F = \mathbb{E}_{\pi_\theta (s, a)} [\nabla \log \pi_\theta (s, a) \nabla \log \pi_\theta (s,a)^T]
$$

Importance sampling:

$$
J(\theta) = \mathbb{E}_{a \sim \pi_\theta} [r(s,a)] = \mathbb{E}_{a \sim \hat{\pi}} [\frac{\pi_\theta (s, a)}{\hat{\pi} (s, a)} r(s, a)]
$$

Increasing the Robustness with Trust Regions:

$$
J_{\theta_{old}}(\theta) = \mathbb{E}_t [\frac{\pi_\theta (a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}R_t] \\
subject \ to \ KL(\pi_{\theta_{old}}(\cdot | s_t) || \pi_\theta(\cdot | s_t)) \leq \delta
$$