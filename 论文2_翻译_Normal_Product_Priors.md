# 论文 2 翻译：Bayesian Compressive Sensing Using Normal Product Priors
**（基于正态乘积先验的贝叶斯压缩感知）**

| 项 | 内容 |
|---|---|
| 作者 | Zhou Zhou, Kaihui Liu, Jun Fang (Member, IEEE) |
| 出处 | IEEE Signal Processing Letters, Vol. 22, No. 5, May 2015, pp. 583–587 |
| DOI | 10.1109/LSP.2014.2364255 |
| 机构 | 电子科技大学，通信抗干扰技术国家级重点实验室（成都 611731） |

> ⚠️ **重要说明（请先读）**：原文公式（尤其是 (3)–(15)）是以嵌入式图形/特殊字体排版，无法从 PDF 中自动提取，本文件中的公式是我根据上下文与标准符号体系**重建**的结果。第 (1)、(2) 式和模型结构可信度很高；**(3)–(15) 的具体解析形式务必在实现前对照原 PDF 逐条核对**。正文的英文叙述部分已完整、忠实翻译。

---

## 摘要

本文提出一种新的促进稀疏性的先验，即"正态乘积"（normal product, NP）先验，并在贝叶斯框架下提出了一种高效的稀疏信号恢复算法。正态乘积分布是两个零均值、方差可能不同的正态变量之积的分布。与其他促进稀疏的分布（如 Student-t 分布）类似，正态乘积分布在原点处具有尖锐峰值，这使其成为鼓励稀疏解的合适先验。本文提出了一个基于正态乘积的两阶段分层模型，并采用变分贝叶斯（VB）方法进行推断。仿真实验表明，与当前最先进的压缩感知算法相比，本文算法在取得相当性能的同时具有更低的计算复杂度。

**关键词**：压缩感知（Compressed Sensing）、正态乘积先验（normal product prior）、稀疏贝叶斯学习（sparse Bayesian learning）。

---

## I. 引言

压缩感知 [1][2] 是一种在过去十年中备受关注的新型数据采集技术。现有压缩感知方法大致可分为以下几类：贪婪追踪法 [3]、凸松弛法 [4] 和非凸优化法 [5]。另一类日益受到关注的压缩感知技术是贝叶斯方法。在贝叶斯框架下，通常给信号赋予一个促进稀疏性的先验，如拉普拉斯（Laplace）先验和高斯-逆伽马（Gaussian-inverse Gamma）先验 [6]，以鼓励稀疏解。一系列实验表明，贝叶斯压缩感知技术 [7] 在稀疏信号恢复方面优于贪婪方法和基追踪（basis pursuit）方法。

贝叶斯压缩感知最常用的先验模型之一是文献 [6] 中提出的高斯-逆伽马先验。高斯-逆伽马先验是一个两层分层模型：第一层为稀疏信号指定高斯先验，第二层为刻画高斯先验的参数指定逆伽马先验。如文献 [8] 所述，这种两阶段分层模型等价于在稀疏信号上施加 Student-t 分布。除高斯-逆伽马先验外，文献 [9] 的作者还采用拉普拉斯先验来促进稀疏解。

本文引入一种新的促进稀疏性先验，即正态乘积（NP）先验，用于稀疏信号恢复。我们建立了一个两阶段正态乘积分层模型，并采用变分贝叶斯（VB）方法对该分层模型进行推断。实验表明，本文算法在取得与稀疏贝叶斯学习（SBL）方法相当性能的同时，具有更低的计算复杂度。

---

## II. 贝叶斯网络

在压缩感知背景下，我们得到某向量带噪声的线性测量：

$$ \mathbf{y} = \boldsymbol{\Phi}\mathbf{x} + \mathbf{w} \tag{1} $$

其中 $\boldsymbol{\Phi}$ 是一个列间低相干性的欠定测量矩阵（$M < N$），$\mathbf{w}$ 表示采集噪声向量，$\mathbf{x}$ 为稀疏信号。对于该逆问题，$\mathbf{x}$ 与噪声均未知，我们的目标是从 $\mathbf{y}$ 中恢复 $\mathbf{x}$。我们将观测噪声建模为零均值独立高斯变量，方差为 $\beta^{-1}$，即

$$ p(\mathbf{w}) = \mathcal{N}\!\left(\mathbf{w};\ \mathbf{0},\ \beta^{-1}\mathbf{I}\right) $$

并寻求 $\mathbf{x}$ 的稀疏估计。

本节为信号采用一个两层分层贝叶斯先验模型。在先验的第一阶段，我们使用正态乘积（NP）分布作为稀疏先验：

$$ \mathbf{x} \sim \mathcal{NP}(\mathbf{0},\ \boldsymbol{\Sigma}),\qquad \boldsymbol{\Sigma} = \mathrm{diag}(\boldsymbol{\gamma}) $$

其中 $\mathbf{0}$ 表示其均值，对角矩阵 $\boldsymbol{\Sigma} = \mathrm{diag}(\boldsymbol{\gamma})$ 表示其协方差矩阵。对 $\mathbf{x}$ 的每个元素，其概率密度函数（PDF）为

$$ p(x_n) = \frac{1}{\pi\sqrt{\gamma_n}}\, K_0\!\left(\frac{|x_n|}{\sqrt{\gamma_n}}\right) \tag{2} $$

该分布在原点处具有尖锐峰值且带有重尾，其中 $K_0$ 是零阶第二类修正贝塞尔函数 [10]。

一个 NP 分布的标量变量可分解为两个独立正态变量之积，即若 $x_n \sim \mathcal{NP}(0,\gamma_n)$，则存在 $u_n \sim \mathcal{N}(0, a_n)$ 与 $v_n \sim \mathcal{N}(0, b_n)$，满足 $x_n = u_n v_n$ 以及矩关系 $\gamma_n = a_n b_n$ [11]。我们称此性质为正态乘积分布的**生成规则**（generating rule）。类似地，对于向量 $\mathbf{x}$，可将其分解为两个相互独立的虚拟正态分布向量 $\mathbf{u}$ 与 $\mathbf{v}$ 的 Hadamard 积，其协方差矩阵分别是对角矩阵 $\mathbf{A} = \mathrm{diag}(\mathbf{a})$ 与 $\mathbf{B} = \mathrm{diag}(\mathbf{b})$，即

$$ \mathbf{u} \sim \mathcal{N}(\mathbf{0},\ \mathbf{A}),\qquad \mathbf{v} \sim \mathcal{N}(\mathbf{0},\ \mathbf{B}),\qquad \mathbf{x} = \mathbf{u} \odot \mathbf{v} $$

其中 $\odot$ 表示 Hadamard 积。最后，在信号模型的第二阶段，我们将 $\mathbf{a}$ 设为伽马超先验的一次实现，并选择 $c \to 0,\ d \to 0$ 以在变分过程中构造更尖锐的 $\mathbf{a}$ 分布 [12]。

根据上述正态乘积的生成规则，两个平行的正态分布变量 $\mathbf{u}$ 和 $\mathbf{v}$ 可作为隐变量引入。将 $\mathbf{x}$ 积分掉后，后验分布可表示为

$$ p(\mathbf{y}\mid\mathbf{u},\mathbf{v}) = \int p(\mathbf{y}\mid\mathbf{x})\,\delta\big(\mathbf{x}-\mathbf{u}\odot\mathbf{v}\big)\, d\mathbf{x} $$

其中 $\delta$ 表示狄拉克（Dirac Delta）函数。在此表达式中可知：在保持 $\mathbf{y}$ 非零的前提下，$\boldsymbol{\Phi}\mathbf{x}$ 的值必须等于 $\boldsymbol{\Phi}(\mathbf{u}\odot\mathbf{v})$。

考虑贝叶斯风险函数

$$ R = \int L\big(\mathbf{x},\ \hat{\mathbf{x}}\big)\, p(\mathbf{x}\mid\mathbf{y})\, d\mathbf{x} $$

其中 $L$ 表示损失函数。因此，在保持贝叶斯风险函数值不变的前提下，我们可在推断过程中用 $\boldsymbol{\Phi}(\mathbf{u}\odot\mathbf{v})$ 替换 $\boldsymbol{\Phi}\mathbf{x}$。于是修正后的贝叶斯网络为：

$$ p(\mathbf{y}\mid\mathbf{u},\mathbf{v}) = \mathcal{N}\!\big(\mathbf{y};\ \boldsymbol{\Phi}(\mathbf{u}\odot\mathbf{v}),\ \beta^{-1}\mathbf{I}\big) \tag{3} $$

$$ p(\mathbf{u}) = \mathcal{N}(\mathbf{u};\ \mathbf{0},\ \mathbf{A}) = \prod_{n} \mathcal{N}(u_n;\ 0,\ a_n) \tag{4} $$

$$ p(\mathbf{v}) = \mathcal{N}(\mathbf{v};\ \mathbf{0},\ \mathbf{B}) = \prod_{n} \mathcal{N}(v_n;\ 0,\ b_n) \tag{5} $$

$$ p(\mathbf{a}) = \prod_{n} \mathrm{Gamma}(a_n;\ c,\ d) \tag{6} $$

如 Fig. 2（贝叶斯网络图）所示。

> **Fig. 1**：三种在相同标准差下的 PDF 对比（用于直观展示 NP 先验在原点处的尖峰）。
> **Fig. 2**：贝叶斯网络结构图。

---

## III. 变分贝叶斯推断

为推断该贝叶斯网络，我们使用平均场变分贝叶斯方法对隐变量的整体后验分布进行解析近似。该方法假设近似变分分布 $q(\mathbf{Z})$（$\mathbf{Z}$ 为模型中的隐变量）可因子化为 $q(\mathbf{Z}) = \prod_k q(\mathbf{Z}_k)$。可证，每个因子 $q(\mathbf{Z}_k)$ 的变分解为

$$ \ln q(\mathbf{Z}_k) = \mathbb{E}_{\mathbf{Z}\setminus\mathbf{Z}_k}\!\Big[\ln p(\mathbf{y}, \mathbf{Z})\Big] + \mathrm{const} $$

其中 $\mathbb{E}_{\mathbf{Z}\setminus\mathbf{Z}_k}[\cdot]$ 表示在除 $\mathbf{Z}_k$ 外的变分分布集合上求期望 [13]。

将变分贝叶斯方法应用于第二节所述贝叶斯模型，$\mathbf{u}$、$\mathbf{v}$、$\mathbf{a}$ 的后验近似分别为：

$$ q(\mathbf{u}) \tag{7} $$
$$ q(\mathbf{v}) \tag{8} $$
$$ q(\mathbf{a}) \tag{9} $$

### A. $\mathbf{u}$ 与 $\mathbf{v}$ 的后验近似

将 (3)(4)(5)(6) 代入 (8)，可得到 $\mathbf{u}$（或 $\mathbf{v}$）的后验近似。从变分推断原理可知，该近似是对应精确后验的逼近；为得到更简洁的迭代公式，我们选择对二次矩做如下松弛：

$$ \langle u_n^2\rangle \approx \hat{u}_n^2 $$

（对 $\mathbf{v}$ 同理：$\langle v_n^2\rangle \approx \hat{v}_n^2$。）

该松弛可解释为在更新中忽略后验方差，其依据是：学习过程结束后后验方差总是趋于零，以保证后验均值集中于估计值。修正后的后验近似为…… 在无噪声情形下，利用 Woodbury 恒等式可得：

$$ \tag{10} $$
$$ \tag{11} $$

类似地，对 $\mathbf{v}$，将 (3)(4)(5)(6) 代入 (7) 可得…… 在无噪声情形下可得：

$$ \tag{12} $$
$$ \tag{13} $$

### B. $\mathbf{a}$ 的后验近似

如第二节所述，将 (4)(5)(6) 代入 (9)，利用 $\mathbf{a}$ 的可分离性，可得…… 即：

$$ \tag{14} $$

类似地有：

$$ \tag{15} $$

---

> ### 📐 公式 (10)–(15) 重建（供参考，实现前务必对照原 PDF）
>
> 由于原 PDF 中 (10)–(15) 的解析式无法自动提取，以下为基于 NP 模型与 SBL/FOCUSS 类 VB 更新规则推导出的**推测形式**。其核心思想：把 $y = \Phi(u\odot v)$ 拆成两个线性模型交替更新 $u$、$v$，并用 $\hat u_n^2$、$\hat v_n^2$ 松弛二次矩。
>
> 记 $\mathrm{diag}(\cdot)$ 为对角阵。噪声为零时，更新 $\mathbf{u}$（视 $\mathbf{v}$ 固定于其估计 $\hat{\mathbf{v}}$）：
>
> $$ \hat{\mathbf{u}} = \mathrm{diag}(\mathbf{a}\odot\hat{\mathbf{v}})\,\boldsymbol{\Phi}^{T}\Big(\boldsymbol{\Phi}\,\mathrm{diag}(\mathbf{a}\odot\hat{\mathbf{v}}^{2})\,\boldsymbol{\Phi}^{T}\Big)^{-1}\mathbf{y} \tag{10·推测} $$
>
> 更新 $\mathbf{v}$（视 $\mathbf{u}$ 固定于 $\hat{\mathbf{u}}$）：
>
> $$ \hat{\mathbf{v}} = \mathrm{diag}(\mathbf{b}\odot\hat{\mathbf{u}})\,\boldsymbol{\Phi}^{T}\Big(\boldsymbol{\Phi}\,\mathrm{diag}(\mathbf{b}\odot\hat{\mathbf{u}}^{2})\,\boldsymbol{\Phi}^{T}\Big)^{-1}\mathbf{y} \tag{12·推测} $$
>
> 两层模型（NP-1）中更新先验精度（$\mathrm{Gamma}$ 超参数）：
>
> $$ a_n = \hat{u}_n^{2} \tag{14·推测} \qquad b_n = \hat{v}_n^{2} \tag{15·推测} $$
>
> 式 (11)、(13) 为 (10)、(12) 的伴随量（协方差/中间项），具体形式需对照原文。**请以原 PDF 中 (10)–(15) 的打印体为准。**

### C. 提出的算法

我们用 "NP-0" 和 "NP-1" 分别表示一层信号模型和两层信号模型的推断过程。NP-0 与 NP-1 的区别在于学习过程是否更新正态乘积先验的精度。

**算法 1：NP-0**

```
输入：y, Φ
输出：x̂
1:  初始化 u, v, … 为常数
2:  while 未收敛 且 t ≤ t_max  do
3:      利用 (12) 更新 …
4:      利用 (10) 更新 …
5:      更新 x̂ = u ⊙ v
6:      t ← t + 1
7:  end while
```

**算法 2：NP-1**

```
输入：y, Φ
输出：x̂
1:  初始化 u, v, a, b, …
2:  while 未收敛 且 t ≤ t_max  do
3:      利用 (12) 更新 u
4:      利用 (13) 更新 v
5:      利用 (10) 更新 …
6:      利用 (11) 更新 …
7:      …
8:      利用 (14) 更新 a
9:      利用 (15) 更新 b
10:     …
11: end while
```

（注：算法伪代码中具体变量名在 PDF 中为公式体，需对照原文确认每行更新的对象。）

NP-0 算法看起来像 FOCUSS 算法 [14]，但二者最根本的区别在于：NP-0 使用 $\mathbf{u}$ 和 $\mathbf{v}$（即 $\mathbf{x}$ 的分解分量）以**交错**方式更新，而 FOCUSS 直接使用整个 $\mathbf{x}$ 更新。第四节实验表明，NP-0 的重构性能与基追踪（BP）[4] 相当，且优于基本 FOCUSS [8]。此外，我们将算法初始值设为常数以避免返回局部最小结果。

在 NP-1 算法中，两个稀疏贝叶斯学习（SBL）[6][8] 过程相互耦合，我们将在第四节通过实验证明 NP-1 的均方误差（MSE）比 SBL 下降更快。

---

## IV. 数值结果

本节将本文算法与 SBL、迭代重加权最小二乘（IRLS）[15] 和 BP 进行比较。在以下结果中，原始信号维度设为 100，每次实验中测量矩阵 $\boldsymbol{\Phi}$ 的每个元素服从标准正态分布。我们忽略观测噪声，仅考虑无噪声情形的算法性能。NP-0 和 NP-1 的停止准则参数 $\epsilon$ 与 $t_{\max}$ 分别设为 $\epsilon$（原文数值）与 300。

图 3 中的成功率定义为成功恢复的平均概率，即成功试验次数与独立实验总次数之比。它汇总了 300 次 Monte Carlo 试验的结果，若 $\lVert \hat{\mathbf{x}} - \mathbf{x}\rVert_2 / \lVert \mathbf{x}\rVert_2 < \epsilon$ 则认为试验成功，其中 $\hat{\mathbf{x}}$ 为原始信号 $\mathbf{x}$ 的估计。图 4 中的 MSE 衡量估计量与真实信号之差的平方的平均值，按下式计算。

图 3 展示了 NP-1 算法在较少测量下的优越性能，给出各算法成功率与测量数 $M$ 及信号稀疏度 $K$ 的关系。可见一层正态乘积算法（NP-0）的重构性能几乎与 BP 相同（BP 是一层拉普拉斯先验框架下未知量的 MAP 估计）。对比还表明，两层正态乘积算法（NP-1）所需 CS 测量数与 SBL 同样少，同时继承了相近的重构精度。

图 4 中，三种算法重构性能的计算开销对比在双核 3.1 GHz CPU、4 GB RAM 的个人 PC 上进行。有趣的是，图 4(a) 中 NP-1 的 MSE 比 SBL 下降更快，同时具有与 SBL 相同的稀疏度-欠采样折中性能。还需注意，我们未使用 [16] 中的基修剪（basis pruning）技术来降低计算复杂度。为理解计算时间低的原因，从图 4(b) 可观察到 NP-1 只需很少几次迭代即达到停止条件。总之，这些实验结果证实 NP-1 是一种快速的稀疏贝叶斯算法。

---

## V. 结论

本文构建了一个基于正态乘积先验的贝叶斯框架，用于解决无噪声情形的压缩感知问题。利用该框架，我们推导了两个算法 NP-0 和 NP-1，并与不同算法进行了比较。我们表明，NP-1 的重构性能与 SBL 相近，而交错更新过程带来了更短的运行时间。

---

## 参考文献（原文保留，附标题翻译）

[1] D. L. Donoho, "Compressed sensing," IEEE Trans. Inf. Theory, vol. 52, no. 4, pp. 1289–1306, 2006.（压缩感知）

[2] E. J. Candès, J. Romberg, and T. Tao, "Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information," IEEE Trans. Inf. Theory, vol. 52, no. 2, pp. 489–509, 2006.（鲁棒不确定性原理：从高度不完整频域信息精确重构信号）

[3] J. A. Tropp and A. C. Gilbert, "Signal recovery from random measurements via orthogonal matching pursuit," IEEE Trans. Inf. Theory, vol. 53, no. 12, pp. 4655–4666, 2007.（通过正交匹配追踪从随机测量中恢复信号）

[4] S. S. Chen, D. L. Donoho, and M. A. Saunders, "Atomic decomposition by basis pursuit," SIAM J. Sci. Comput., vol. 20, no. 1, pp. 33–61, 1998.（基追踪的原子分解）

[5] R. Chartrand, "Exact reconstruction of sparse signals via nonconvex minimization," IEEE Signal Process. Lett., vol. 14, no. 10, pp. 707–710, 2007.（非凸最小化的稀疏信号精确重构）

[6] M. E. Tipping, "Sparse bayesian learning and the relevance vector machine," J. Mach. Learn. Res., vol. 1, pp. 211–244, 2001.（稀疏贝叶斯学习与相关向量机）

[7] S. Ji, Y. Xue, and L. Carin, "Bayesian compressive sensing," IEEE Trans. Signal Process., vol. 56, no. 6, pp. 2346–2356, 2008.（贝叶斯压缩感知）

[8] D. P. Wipf and B. D. Rao, "Sparse bayesian learning for basis selection," IEEE Trans. Signal Process., vol. 52, no. 8, pp. 2153–2164, 2004.（用于基选择的稀疏贝叶斯学习）

[9] S. D. Babacan, R. Molina, and A. K. Katsaggelos, "Bayesian compressive sensing using laplace priors," IEEE Trans. Image Process., vol. 19, no. 1, pp. 53–63, 2010.（拉普拉斯先验的贝叶斯压缩感知）

[10] E. W. Weisstein, "Normal product distribution. MathWorld–A wolfram web resource." [Online].（正态乘积分布）

[11] A. Seijas-Macías and A. Oliveira, "An approach to distribution of the product of two normal variables," Discuss. Math.: Probabil. Statist., vol. 32, 2012.（两正态变量之积分布的一种方法）

[12] D. Wipf, J. Palmer, and B. Rao, "Perspectives on sparse bayesian learning," Comput. Eng., vol. 16, no. 1, p. 249, 2004.（稀疏贝叶斯学习之视角）

[13] D. G. Tzikas, C. Likas, and N. P. Galatsanos, "The variational approximation for bayesian inference," IEEE Signal Process. Mag., vol. 25, no. 6, pp. 131–146, 2008.（贝叶斯推断的变分近似）

[14] I. F. Gorodnitsky and B. D. Rao, "Sparse signal reconstruction from limited data using focuss: A re-weighted minimum norm algorithm," IEEE Trans. Signal Process., vol. 45, no. 3, pp. 600–616, 1997.（FOCUSS：一种重加权最小范数算法的有限数据稀疏信号重构）

[15] R. Chartrand and W. Yin, "Iteratively reweighted algorithms for compressive sensing," in Proc. ICASSP, 2008, pp. 3869–3872.（压缩感知的迭代重加权算法）

[16] M. E. Tipping and A. C. Faul et al., "Fast marginal likelihood maximisation for sparse bayesian models," in Proc. 9th Int. Workshop on AIStats, 2003, vol. 1, no. 3.（稀疏贝叶斯模型的快速边缘似然最大化）
