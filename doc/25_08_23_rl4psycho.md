# 1 Tasks and Goals

**Problem.** Given logs of many participants’ **interaction trajectories** (state–action–next state), we seek to:

* Obtain an **interpretable individual score** $\beta_j>0$ (participant $j$’s “decisiveness/ability”).
* Learn a generalizable **task value function** $Q_\theta(s,a)$ and, optionally, a **reward function** $r_w(s,a,s')$ (learning $r$ yields an inverse RL setting).
* Ensure the model both **explains observed behavior** (predicts next-action distributions) and **respects task dynamics** (values do not collapse into arbitrary, overfit functions).

**Core idea.** Use an **MDP** to represent the task (“objective value”) and a **maximum-entropy / softmax** policy to capture decision making (“how people act given value”). Individual differences are captured by a single, interpretable **temperature** $\beta_j$. All parameters are estimated jointly via **penalized maximum likelihood / generalized EM (GEM)**.

---

# 2 Notation

* MDP: $(\mathcal S,\mathcal A,T,R,\gamma)$.
* **Soft Bellman temperature** $\tau>0$ (backup only; fixes scale; typically $\tau=1$):
  $$
  V_\theta(s)=\tau\log\sum_{a}\exp\!\Big(\tfrac{1}{\tau}Q_\theta(s,a)\Big).
  $$
* **Soft Bellman residual**:
  $$
  \delta(s,a,s')=Q_\theta(s,a)-\big(r_w(s,a,s')+\gamma\,V_\theta(s')\big).
  $$
* **Policy**:
  $$
  \pi_\theta(a\mid s,\beta_j)=\frac{\exp\{\beta_j\,Q_\theta(s,a)\}}{\sum_{a'}\exp\{\beta_j\,Q_\theta(s,a')\}}.
  $$
* **Ability prior**: $\log\beta_j\sim\mathcal N(\mu,\sigma^2)$.
* **Reward head bounded**: $r_w(\cdot)\in[-1,1]$ (e.g., via $\tanh$ or an L2 magnitude constraint).

---

# 3 Objective: Penalized Observed Log-Likelihood

$$
\begin{aligned}
\mathcal J(\theta,w,\{\beta_j\})
&=\underbrace{\Big[-\sum_{j}\sum_{(s,a)\in\mathcal D_j}\log\pi_\theta(a\mid s,\beta_j)\Big]}_{\mathcal L_{\text{beh}}:\ \text{Behavior NLL}}\\
&\quad+\ \lambda_{\text{bell}}\ \underbrace{\mathbb E_{(s,a,s')\sim D}\big[\delta^2\big]}_{\mathcal L_{\text{bell}}:\ \text{Soft Bellman consistency}}\\
&\quad+\ \lambda_{\text{cql}}\ \underbrace{\Big(\mathbb E_s\log\!\sum_a e^{Q_\theta(s,a)}-\mathbb E_{(s,a)\sim D} Q_\theta(s,a)\Big)}_{\mathcal L_{\text{cql}}:\ \text{CQL regularizer}}\\
&\quad+\ \underbrace{\sum_j \mathrm{KL}\!\big((\log\beta_j)\,\|\,\mathcal N(\mu,\sigma^2)\big)}_{\text{Prior / shrinkage on }\beta}\\
&\quad+\ \underbrace{\mathcal R_r(w)}_{\text{Reward magnitude/L2}}.
\end{aligned}
$$

Intuition:

* $\mathcal L_{\text{beh}}$ pulls $Q$ into shapes that **explain choices**.
* $\mathcal L_{\text{bell}}$ (+ $\mathcal L_{\text{cql}}$) pulls $Q$ back toward **dynamics-consistent** solutions.
* The prior shrinks $\beta$ for stability and empirical Bayes regularization.

> $Q$ neither drifts from dynamics nor overfits noise; **$\beta$** remains dedicated to individual decisiveness/randomness.

---

# 4 Identification Issues

*To be completed.* (e.g., temperature vs. value scale, translation/scale of $Q$, state-wise centering techniques, and constraints that prevent $\beta$–$Q$ co-scaling.)

---

# 5 Generalized EM: Alternating Optimization

Alternate **individual** ($\beta$) and **shared** ($Q_\theta, r_w$) updates.

## 5.1 E-step: Update $\beta_j$ (per participant)

Let $z_j=\log\beta_j$. Given current $Q,r$, maximize
$$
\ell_j(z)=\sum_{(s,a)\in\mathcal D_j}\!\Big[e^z Q(s,a)-\log\!\sum_{a'}e^{e^z Q(s,a')}\Big]-\frac{(z-\mu)^2}{2\sigma^2}.
$$

**Gradient/Hessian** (concave in $z$; 1D Newton, ~5–10 steps; participants are independent/parallelizable):
$$
\begin{aligned}
g_j &= e^z\sum_t\big[Q_t(a_t)-\mathbb E_{\pi_\beta}[Q_t(a)]\big]-\frac{z-\mu}{\sigma^2},\\
H_j &= -e^{2z}\sum_t \mathrm{Var}_{\pi_\beta}[Q_t(a)] + e^z\sum_t\big[Q_t(a_t)-\mathbb E_{\pi_\beta}[Q_t]\big]-\frac{1}{\sigma^2}.
\end{aligned}
$$

Newton step $z\leftarrow z-g_j/H_j$, then set $\hat\beta_j=e^z$.

## 5.2 M-step: Update $Q_\theta, r_w$ (shared)

Fix $\hat\beta$; run a few small-step SGD iterations to minimize
$$
\mathcal L_{\text{beh}}\ \text{(or }\mathcal L_{\text{beh-mix}}\text{)}
+\lambda_{\text{bell}}\mathcal L_{\text{bell}}
+\lambda_{\text{cql}}\mathcal L_{\text{cql}}
+\mathcal R_r.
$$

Guidelines:

* Use **target networks** and stable **log-sum-exp** backups.
* Prefer **CQL** or light **behavior cloning** for offline stability.
* After each round, check (penalized) and validation NLL; if not improved, **rollback** and reduce LR to maintain GEM’s non-increasing property.

## 5.3 Mini-batch / Online EM

Per outer iteration:

1. Sample a small set of participants $U$; run parallel E-steps to update/cache $\hat\beta_j$.
2. Use their transitions for a few M-step SGD updates.


---

# 6 Evaluation and Uncertainty

## 6.1 Held-out NLL

1. Split by participant ID into train/valid/test.
2. Train on train; estimate each test participant’s $\hat\beta_j$ using only their training segment (or priors/encoder)—**never** peek at test segments.
3. For each test $(s,a)$ compute
   $$
   \pi(a\mid s,\hat\beta_j)=\frac{\exp\{\hat\beta_j\,Q(s,a)\}}{\sum_{a'}\exp\{\hat\beta_j\,Q(s,a')\}}.
   $$
4. Aggregate:
   $$
   \text{Held-out NLL}=-\frac{1}{M}\sum_{(s,a)\in\text{test}}\log\pi(a\mid s,\hat\beta_j).
   $$

## 6.2 External Validity

Treat $\hat\beta_j$ as **decisiveness/ability**. For an external criterion $Y_j$ (post-test, performance, ratings).

* **Pearson**:
  $$
  r=\frac{\sum_j (\hat\beta_j-\bar\beta)(Y_j-\bar Y)}{\sqrt{\sum_j (\hat\beta_j-\bar\beta)^2}\sqrt{\sum_j (Y_j-\bar Y)^2}}.
  $$
* **Spearman**: rank-transform $\hat\beta,Y$ then compute Pearson.

## 6.3 Confidence Intervals

With $Q$ fixed, define the participant-level log posterior
$$
\ell(\beta)=\sum_t\!\Big[\beta Q(s_t,a_t)-\log\sum_a e^{\beta Q(s_t,a)}\Big]+\log p(\beta).
$$
At $\hat\beta$, use a second-order Taylor expansion. Observed information $J(\hat\beta)=-\ell''(\hat\beta)$ gives
$$
\text{SE}(\hat\beta)\approx \frac{1}{\sqrt{J(\hat\beta)}}\quad\text{and}\quad 95\%\ \text{CI}:\ \hat\beta\pm1.96\,\text{SE}.
$$

> Prefer the log scale $z=\log\beta$ for stronger concavity:
> 1) Newton to get $\hat z$ and Hessian $H$.
> 2) $\text{SE}_z\approx\sqrt{(-H)^{-1}}$.
> 3) Log-scale CI $[\hat z\pm1.96\,\text{SE}_z]$, then exponentiate:
> $[e^{\hat z-1.96\,\text{SE}_z},\,e^{\hat z+1.96\,\text{SE}_z}]$.

