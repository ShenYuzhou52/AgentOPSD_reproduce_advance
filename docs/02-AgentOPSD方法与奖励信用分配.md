# AgentOPSD 方法与奖励信用分配（关键技术设计）

> 本文是项目最核心的技术文档。它解释：为什么 RLVR 的轨迹级奖励不够用、AgentOPSD 如何把稀疏的结果监督变成稠密的 turn 级信用、每一步的数学动机、以及本仓库如何把它落到代码里。

## 1. 问题设定：RLVR 在长程智能体任务中的信用分配困境

### 1.1 符号

任务 $x$，初始观测 $o_0$。智能体第 $k$ 轮根据可见历史 $s_k=(x,o_0,a_1,o_1,\dots,a_{k-1},o_{k-1})$ 采样动作

$$a_k=(y_{k,1},\dots,y_{k,L_k})\sim\pi_\theta(\cdot\mid s_k),\qquad y_{k,t}\text{ 是第 }t\text{ 个 token。}$$

一个 $K$ 轮轨迹 $\tau=(s_1,a_1,o_1,\dots,s_K,a_K,o_K)$ 结束时环境给出**二值结果** $R(\tau)\in\{0,1\}$（可验证奖励）。

### 1.2 GRPO：轨迹级优势的"均匀广播"

GRPO 对每个任务采样 $G$ 条轨迹，计算序列级优势：

$$A^{(i)}_{\text{seq}}=\frac{R^{(i)}-\bar R}{\hat\sigma_R+\epsilon_0},\qquad \bar R=\frac1G\sum_{j=1}^{G}R^{(j)}.$$

然后**把同一个 $A_{\text{seq}}$ 广播给轨迹里的每一个 token**。

问题：

- 成功轨迹里可能有**虚假、冗余甚至误导**的动作；失败轨迹里也可能有**有用的推理**。均匀广播无法区分。
- 任务越长，$A_{\text{seq}}$ 覆盖的决策越多，均匀广播越失真——这正是论文 Figure 1(b) 里"每多一轮就掉更多分"的原因。
- 理论上（论文 Proposition 6），仅凭轨迹回报无法识别每个 turn 的贡献：两条回报相同的轨迹，一条靠单个决定性决策成功，另一条靠均匀推进成功，但 GRPO 给它们完全相同的逐 token 更新。

### 1.3 已有方案的不足

**特权自蒸馏（OPSD 族）** 提供了更稠密的 token 级信号：同一个策略对同一段学生生成，用带"训练时才可见的特权信息（技能 c⁺）"的 teacher 分支重新打分，得到 teacher–student gap：

$$\delta_{k,t}=\log\pi_\theta(y_{k,t}\mid h^+_{k,t})-\log\pi_\theta(y_{k,t}\mid h_{k,t}).$$

但这个局部 gap 有两个错配：

1. **粒度错配**：智能体环境只在 turn 边界给反馈，多个 token 共同构成一个动作；token 级 gap 与交互节奏不对齐。
2. **时序错配**：即使把 gap 聚合到 turn（StepOPSD 的做法），也是"每轮孤立打分"，没有考虑此前轮次已经积累的证据——同一个 gap 在开局时可能是决定性的，在结局已定时就是冗余的。

> 一句话总结 AgentOPSD 的核心观点：**局部 self-distillation gap 本身不是顺序信用；信用是它"对最终成功估计的边际修订量"。**

## 2. AgentOPSD 三步法

### 2.1 第 1 步：把 token 级 gap 聚合成 turn 级证据

$$e_k=\sum_{t=1}^{L_k}\delta_{k,t}=\log\frac{\pi_\theta(a_k\mid s_k,c^+)}{\pi_\theta(a_k\mid s_k)}.$$

$e_k>0$ 表示技能（成功相关的先验行为）让这轮动作更可能；$e_k<0$ 相反。论文把它解释为贝叶斯证据的**可计算代理**：理想证据是成功条件分布与失败条件分布的对数似然比（Bayes factor），在"teacher 分支近似成功条件分布、且成功较稀有时期望分布近似失败条件分布"的假设下（附录 A.1），$e_k$ 与理想 Bayes factor 同号且保序。

### 2.2 第 2 步：log-odds 空间的递归贝叶斯信念更新

把"轨迹最终成功"看作一个未知事件 $C$，维护信念 $B_k=\Pr(C\mid s_k,a_k)$。用组成功率作为先验：

$$B_0=\operatorname{clip}\!\big(\bar R,\ \epsilon_0,\ 1-\epsilon_0\big),\qquad \epsilon_0=10^{-3},\qquad \ell_0=\operatorname{logit}(B_0).$$

每轮把证据衰减地累加进 log-odds：

$$c_k=\gamma\,c_{k-1}+e_k,\qquad \ell_k=\ell_0+c_k,\qquad B_k=\sigma(\ell_k).$$

关键量是**边际信念修订**：

$$\Delta B_k=B_k-B_{k-1}\ \approx\ B_{k-1}(1-B_{k-1})\cdot\big(e_k-(1-\gamma)c_{k-1}\big).$$

直觉：

- $\gamma<1$ 让老证据按几何衰减，避免长轨迹里早期证据把信念"钉死"；$\gamma=1$ 退化为经典序贯检验里的对数似然比累积（Wald）。
- $B(1-B)$ 是信念敏感性门：结果最不确定（$B\approx0.5$）时证据影响最大；信念接近 0 或 1 时证据被抑制。
- 所以**同一个 $e_k$，在证据还没积累起来时是决定性的，在信念已经确定时几乎不影响更新**——这就是"历史依赖的顺序信用"。

### 2.3 第 3 步：有界优势整形

先让修订方向与验证器结果对齐：

$$q_k=\operatorname{sign}(A_{\text{seq}})\cdot\Delta B_k.$$

$\Delta B_k$ 只看"改了多少"，$q_k$ 再看"改的方向是否与最终结果一致"：成功轨迹里向上的修订（$A_{\text{seq}}>0$）获得正信用，失败轨迹里同样的向上修订获得负信用。

然后在轨迹内做标准化并施加有界乘子：

$$z_k=\frac{q_k-\mu_q}{\sigma_q+\epsilon},\qquad w_k=\operatorname{clip}\big(1+b\,z_k,\ 1-b,\ 1+b\big),\qquad b\in(0,1),$$

$$\tilde A_k=A_{\text{seq}}\cdot\big((1-\lambda)+\lambda\,w_k\big),\qquad \lambda\in[0,1].$$

每个 token 继承它所属轮次的 $\tilde A_k$，目标函数就是标准的带裁剪 GRPO 目标，只是把 $A$ 换成 $\tilde A$：

$$\mathcal{L}_{\text{AgentOPSD}}(\theta)=\frac1G\sum_i\frac{1}{\sum_t M_{i,t}}\sum_t M_{i,t}\ \min\!\Big(r_{i,t}\tilde A^{(i)}_{\kappa_i(t)},\ \operatorname{clip}(r_{i,t},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})\tilde A^{(i)}_{\kappa_i(t)}\Big)+\beta_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}.$$

其中 $M$ 是响应 token 掩码，$\kappa_i(t)$ 把 token 映射到轮次，$r_{i,t}$ 是 importance ratio。**没有额外蒸馏损失**：teacher 信号只通过 $\tilde A$ 起作用。

## 3. 算法伪代码（论文 Algorithm 1）

```text
输入: 策略 π_θ, 验证器 R, 组大小 G, 技能检索器, λ, b, γ
每个训练 step:
  采样一批任务 {x}
  对每个任务 x（带检索技能 c⁺）:
    采样 G 条轨迹 {τ^(1)...τ^(G)} ~ π_θ(·|x)          # 在线 rollout
    对每条轨迹 i: R^(i) = R(x, τ^(i)) ∈ {0,1}
    A^(i)_seq = (R^(i) − R̄) / (σ̂_R + ε0)             # 组相对优势
    对每条轨迹 i:
      B0 = clip(S/G, ε0, 1−ε0);  ℓ0 = logit(B0);  c0 = 0
      for k = 1..K_i:                                   # 逐轮
        e_k = Σ_t sg[log π_θ(y_t|s⁺_k) − log π_θ(y_t|s_k)]   # 一次 teacher 前向
        c_k = γ c_{k−1} + e_k;  ℓ_k = ℓ0 + c_k
        B_k = σ(ℓ_k);          ΔB_k = B_k − B_{k−1}
      q_k = sign(A^(i)_seq)·ΔB_k                        # 结果对齐
      z_k = (q_k − μ_q)/(σ_q + ε);  w_k = clip(1+b z_k, 1−b, 1+b)
      Ã_k = A^(i)_seq·((1−λ) + λ w_k)                  # 有界整形
      token 继承所属轮次的 Ã_k
  用裁剪 GRPO 目标更新 θ
```

成本：相比 GRPO 只多 **每条轨迹一次 teacher 前向**（SDAR 本来就要算），无 critic、无额外 rollout。

## 4. 与相关方法的关系

| 方法 | 信用粒度 | 信号来源 | 关键差异 |
|---|---|---|---|
| GRPO | 轨迹级 | 组相对结果 | 均匀广播，无 turn 级信用 |
| OPSD | token 级 | 蒸馏 gap 作为额外目标 | 与 turn 边界不对齐 |
| GRPO+OPSD | 轨迹级 + token 级 | 结果 + 蒸馏目标 | 两者并列，gap 不参与信用分配 |
| RLSD | token 级 | gap 缩放优势 | 无历史依赖，仅局部缩放 |
| Skill-SD / SDAR | token 级 | gap 门控蒸馏损失 | GRPO 优势不变，gap 作辅助损失 |
| StepOPSD | turn 级 | turn 内 gap 之和 | 每轮孤立打分，无顺序累积 |
| **AgentOPSD（本仓库）** | **turn 级** | **递归信念修订 ΔB 整形优势** | 历史依赖 + 结果对齐 + 有界整形 |

## 5. 关键设计点（为什么这样设计）

1. **局部 gap ≠ 顺序信用**：消融（论文 Table 2）里把递归修订换成原始 gap，ALFWorld 从 89.1% 掉到 82.8%。信用必须由"对累计信念的修订"定义。
2. **turn 边界聚合**：token 级累积把单个决策切碎，降到 85.9%；环境在 turn 边界反馈，信用也应该在 turn 边界结算。
3. **结果对齐的符号**：只看 $|\Delta B|$ 会掉到 80.5%——因为同样的修订在成功/失败轨迹里含义相反。`sign(A_seq)` 提供了方向。
4. **组成功率先验 B0**：去掉先验锚点掉到 78.9%。B0 同时是"任务难度"的验证器接地估计和 $B(1-B)$ 门的操作区间。
5. **有界性与保号性**：$w_k\in[1-b,1+b]$、$((1-\lambda)+\lambda w_k)>0$，所以 $\tilde A$ 与 $A_{\text{seq}}$ 同号且大小有界（Proposition 1-2），不会翻转 GRPO 的更新方向，也不会放大异常优势；$\lambda=0$ 时精确恢复 GRPO（Proposition 3）。
6. **超参共享**：论文用单一配置 λ=0.5、b=0.2、γ=0.95、ε0=1e-3、clip 0.2/0.24、KL 0.01、LR 1e-6、组大小 8，跨 3 个环境和 2 个规模不做逐任务调参；敏感性只有 λ 有系统性影响。

## 6. 本仓库的实现细节

### 6.1 数据形状

verl-agent/SDAR 的多轮 rollout 把**每一轮**作为 batch 的一行：

- `traj_uid`：行属于哪条轨迹；
- `turn_step`：行在轨迹内的轮次序号（只用来排序）；
- `episode_rewards`：该轨迹的最终结果（同一轨迹的每行重复）；
- `uid`：GRPO 组（任务）id；
- `teacher_log_probs` / `old_log_probs`：同一段生成的 teacher / student token 级对数概率（teacher 已 detach）；
- `response_mask`：有效响应 token；
- `advantages`：GRPO 广播后的轨迹级优势（reshape 前每行所有 token 相同）。

### 6.2 实现位置

- `agentopsd/credit.py::reshape_advantages`：纯 PyTorch 实现 2.1–2.3 全部公式，输入上述张量/数组，输出整形后的优势 + 一组 `agentopsd/*` 诊断指标（证据均值、信念修订均值、乘子范围、pivotal turn 占比、整形前后优势 std 等）。
- `agentopsd/trainer/patch.py::install`：包装 SDAR `SkillSDRayTrainer.fit` 里调用的 `compute_advantage`——GRPO 优势算完后立刻做信用整形，再交给 actor 更新。改动面只有这一个函数。
- `agentopsd/trainer/main_agentopsd.py`：训练入口，等价于 SDAR 的 `main_skillsd`（teacher 前向、SkillProvider、环境 worker 全部复用），但关闭 SDL 蒸馏损失，并安装上面的钩子。
- 超参通过 Hydra 传入：`+algorithm.agentopsd.{enable,lam,b,gamma,eps0,success_threshold,skills_dir,skill_all}`。

### 6.3 正确性保障

- `scripts/test_credit.py` 覆盖：λ=0 精确恢复 GRPO、单轮轨迹恢复 GRPO、符号保持、乘子有界、正证据轮次获得更大权重、全同组零优势、配置解析。
- 组成功率先验按 `episode_rewards > success_threshold` 计算（ALFWorld 奖励为 0/10，阈值 0.5 得到二进制成功）。
- 每组/每轨迹按 `uid`/`traj_uid` 分组、`turn_step` 排序，避免批次乱序影响递归。

## 7. 论文结果参考（复现目标）

| 模型 | 方法 | ALFWorld 平均成功率 |
|---|---|---|
| Qwen2.5-3B-Instruct | GRPO | 75.0 |
| Qwen2.5-3B-Instruct | AgentOPSD | 84.4 |
| Qwen2.5-7B-Instruct | GRPO | 81.2 |
| Qwen2.5-7B-Instruct | AgentOPSD | **89.1** |

本项目用 8B（Qwen3-8B-Instruct）做稳定性实测；如需严格对照论文数值，可用 `MODEL_NAME=Qwen/Qwen2.5-7B-Instruct` 复跑。

