- $\Gamma$点非0频率有3个简并解 1LO 2TO
  
在光子晶体中（周期介质）波的形式是布洛赫波：$\mathbf{H}_{\mathbf{k}}(\mathbf{r}) = \mathbf{u}_{\mathbf{k}}(\mathbf{r}) e^{i\mathbf{k}\cdot\mathbf{r}}$。其中$\mathbf{u}_{\mathbf{k}}(\mathbf{r})$是一个周期函数，它在晶胞内部剧烈变化。将它代入$\nabla \cdot \mathbf{H} = 0$，利用矢量恒等式，我们得到：
$$ \nabla \cdot (e^{i\mathbf{k}\cdot\mathbf{r}} \mathbf{u}_{\mathbf{k}}) = e^{i\mathbf{k}\cdot\mathbf{r}} (i\mathbf{k} \cdot \mathbf{u}_{\mathbf{k}} + \nabla \cdot \mathbf{u}_{\mathbf{k}}) = 0 $$

消去指数项，真正的约束条件变成了：
$$ik\cdot u_k(r)+\nabla\cdot u_k(r)=0$$

请仔细看这个公式，这就是第 3 个解“复活”的秘密：

即使宏观上场的包络方向（$\mathbf{u}_{\mathbf{k}}$的平均方向）与$\mathbf{k}$平行（看起来像纵波），只要$\mathbf{u}{\mathbf{k}}$在晶胞内部的微观变化（$ \nabla \cdot \mathbf{u}{\mathbf{k}} $）能够产生一个相反的值来抵消 $ i\mathbf{k} \cdot \mathbf{u}{\mathbf{k}}$，那么总的散度依然可以是 0，类似声子晶体每支光学支有3个模(2 TO; 1 LO)

结论： 晶格的微观结构提供了一个额外的自由度（$\nabla \cdot \mathbf{u}_{\mathbf{k}}$），使得宏观上看起来像“纵模”的波，在微观上依然满足无源条件。因此，第 3 个解是物理上允许存在的。
- $\Gamma$点0频率仅有2个简并解 2 TA

$\Gamma$点因为在$\omega \rightarrow 0,k \rightarrow 0$时，波长 $\lambda \rightarrow \infty$。光子晶体表现得像一个均匀的有效介质，此时光波“看不见”微观结构，$u_k(r)$ 变成了一个常数（不再剧烈变化）。当 $u_k$是常数时，$\nabla \cdot u_k =0$，光波必须满足均匀介质中的麦克斯韦方程，仅有2个横波解。

---
$$\nabla \times (\frac{1}{\varepsilon(r)}\nabla \times H) = \frac{\omega^2}{c^2}H$$
when H is static , here I mean $j_f=\frac{\partial D}{\partial t}=0, \nabla \times H=0$, we have $\omega=0$, this solution always exists, meaning static field and no energy flow.

$∇·{ ∇ × [ (1/ε) ∇ × H ] } = (ω/c)^2 ∇·H$  $\rightarrow\nabla\cdot H=0$ if $\omega\neq 0 $ this is transverse condition , giving us only 2 independent solutions of $\vec{H}$

at $\Gamma$ point , $k=0$, $\vec{H}_k=e^{ikr}\vec{u}_k(r)$ $\quad \nabla \cdot \vec{H}=0$ gives $\nabla \cdot \vec{u}_k(r)=0 $

二、为什么 k≠0 时只有两条带连出 Γ 点？
因为对任意 k≠0、ω≠0 的本征解，∇·H=0 约束在每一个平面波分量上变成 (k+G)·h_G=0。对每个 G，h_G 被约束在与 (k+G) 垂直的二维子空间里。对最低两条“声学支”，主导分量是 G=0，于是 k·h_0=0，只有两个独立偏振。于是与 Γ 点相连的传播带只有两条。这一点与介质对比度大小无关，是由算符结构和傅里叶展开的几何约束决定的。
三、那为什么在 Γ 点（k=0，ω≠0）会看到“三重简并”？
关键区分：
“只有两条与 ω=0 连出的传播带”是指在 Γ 点附近的低频线性色散（声学支）起源于 G=0 的基波；它只有两个偏振。
“Γ 点有三重简并”指的是在更高频的驻波本征态（光学支）处，由晶体的点群（例如 Oh）允许的不可约表示（如 T1g/T2g）带来的简并。这里的三重并不是“第三个传播自由度”，而是不同空间分布的三个无散模在该点能量相同。它们依旧满足 ∇·u=0，只是三种互相正交的模式在对称性下等能。
四、k=0, ω=0 点为何 ill-defined
在 ω=0 时，上面的推导不再强制 ∇·H=0；同时 ∇×H=0 的纵向场 H=∇ψ 也是本征解（本征值0）。因此零频处解空间混入无限多静场（无旋但不一定无散），和两条传播分支的极限在此点“撞车”，导致方向取向依赖逼近路径，故称 ill-defined。
一旦 ω>0，再回到 Γ 点，主方程强制 ∇·u=0，驻波解是良好定义的，并可按点群不可约表示分类。

---
little group:  gk = k+G
supergroup: ?
subgroup: ?
symmorphic group: keep one common point fixed, like only contains a point group operation

---
B. 二次项 (Quadratic term in q 2)
二次项有两个来源：
展开式一阶部分中的 δE (2)。
展开式二阶部分中的 (δE(1)) 2。
$$ \delta \Omega_{q^2} = \underbrace{\sum_{\mathbf{k}} -(1 - 2f(E_{\mathbf{k}})) \frac{\hbar^2 q^2}{8m} \frac{\xi_{\mathbf{k}}}{E_{\mathbf{k}}}}_{\text{项 1}} + \underbrace{\sum_{\mathbf{k}} \frac{\partial f}{\partial E_{\mathbf{k}}} \left( \frac{\hbar^2 (\mathbf{k} \cdot \mathbf{q})}{2m} \right)^2}_{\text{项 2}}$$


**分析项 1 (密度项/反磁项):**
这一项可以理解为化学势的重整化。注意到 $\xi_{\mathbf{k}+\mathbf{q}/2} \approx \xi_{\mathbf{k}} + \mathbf{v}\cdot\mathbf{q} + \frac{\hbar^2 q^2}{8m}$。最后这个常数项 $\frac{\hbar^2 q^2}{8m}$ 相当于 $\xi_{\mathbf{k}}$ 整体平移，即化学势 $\mu$ 减小了 $\delta \mu = \frac{\hbar^2 q^2}{8m}$。
根据热力学关系 $\frac{\partial \Omega}{\partial \mu} = -N$（$N$ 为总粒子数），因此 $\delta \Omega \approx -N (-\delta \mu) = N \frac{\hbar^2 q^2}{8m}$。
更严格的推导利用粒子数方程：$N = \sum_{\mathbf{k}} [u_{\mathbf{k}}^2 f + v_{\mathbf{k}}^2 (1-f)] = \frac{1}{2} \sum_{\mathbf{k}} (1 - \frac{\xi_{\mathbf{k}}}{E_{\mathbf{k}}} \tanh(\frac{\beta E_{\mathbf{k}}}{2}))$。
项 1 中的求和正是：
$$-\frac{\hbar^2 q^2}{8m} \sum_{\mathbf{k}} \frac{\xi_{\mathbf{k}}}{E_{\mathbf{k}}} (1-2f) = -\frac{\hbar^2 q^2}{8m} \sum_{\mathbf{k}} \frac{\xi_{\mathbf{k}}}{E_{\mathbf{k}}} \tanh\left(\frac{\beta E_{\mathbf{k}}}{2}\right)$$

利用上述粒子数关系（忽略常数真空项），该项贡献正比于总粒子数 $N$：
$\text{项 1} \approx N \frac{\hbar^2 q^2}{8m}$

**分析项2(电流响应项/顺磁项):**
$$\text{项 2} = \frac{\hbar^4 q^2}{4m^2} \sum_{\mathbf{k}} (\hat{\mathbf{k}} \cdot \hat{\mathbf{q}})^2 k^2 \frac{\partial f}{\partial E_{\mathbf{k}}}$$

在各向同性系统中，$\langle (\hat{\mathbf{k}} \cdot \hat{\mathbf{q}})^2 \rangle = 1/3$。
$$\text{项 2} = \frac{\hbar^2 q^2}{8m} \left[ \frac{2}{3} \sum_{\mathbf{k}} \frac{\hbar^2 k^2}{2m} \frac{\partial f}{\partial E_{\mathbf{k}}} \right]$$

方括号内的部分正是朗道公式中定义的 **正常流体normal fluid**粒子数 $-N_n$。
$$\text{项 2} = - N_n \frac{\hbar^2 q^2}{8m}$$

**合并两项:**
$$\delta \Omega = \text{项 1} + \text{项 2} = (N - N_n) \frac{\hbar^2 q^2}{8m}$$

根据二流体模型，超流电子数 $N_s = N - N_n$。
所以：
$$\delta \Omega = N_s \frac{\hbar^2 q^2}{8m}$$

为了匹配题目中的公式(1.5)，我们将上式变形：
$$\delta \Omega = \frac{\hbar^2 q^2}{4m} \frac{N_s}{2}$$

题目中的 $n_s(T)$ 指的是**超流电子的总数** $N_s$（或者题目中的 $\delta \Omega$ 是自由能密度，而 $n_s(T)$ 是粒子数密度，公式中的 $V$ 可能位置有误或定义不同，但物理形式是一致的）。
如果按照题目公式 (1.5) 的写法 $\frac{n_s(T)}{2V}$，且 $\delta \Omega$ 为自由能密度，则 $n_s(T)$ 为总粒子数。

最终得到：
$$\delta \Omega = \frac{\hbar^2 q^2}{4m} \frac{n_s(T)}{2}V + O(q^4) $$


---

### 2. String 算符的物理图像：非局域的“尾巴”

Jordan-Wigner 变换的核心定义如下（以自旋 $S=1/2$ 的升降算符为例）：

$$ \sigma_j^+ = \left( e^{i\pi \sum_{l<j} n_l} \right) c_j^\dagger $$
$$ \sigma_j^- = \left( e^{-i\pi \sum_{l<j} n_l} \right) c_j $$
$$ \sigma_j^z = 2n_j - 1 $$

其中，括号里的部分 $e^{i\pi \sum_{l<j} n_l}$ 就是所谓的 **String 算符**（弦算符），通常记作 $\phi_j$ 或 $K_j$。

#### **(1) 它的数学含义是什么？**

让我们拆解一下这个指数项。根据欧拉公式 $e^{i\pi} = -1$，我们可以把这个算符理解为对粒子数的奇偶性进行计数：

$$ e^{i\pi n_l} = (-1)^{n_l} $$

*   当格点 $l$ **没有粒子** ($n_l=0$) 时，这一项等于 **$+1$**。
*   当格点 $l$ **有粒子** ($n_l=1$) 时，这一项等于 **$-1$**。

因此，整个 String 算符实际上是一个连乘积：

$$ \text{String}_j = \prod_{l=1}^{j-1} (-1)^{n_l} $$

**它的物理意义非常直观：**
它在数第 $j$ 个格点左侧（所有 $l < j$ 的位置）总共有多少个费米子。
*   如果左边有 **偶数** 个费米子，String 算符的值为 **$+1$**（不做改变）。
*   如果左边有 **奇数** 个费米子，String 算符的值为 **$-1$**（添加一个负号相位）。

#### **(2) 形象的物理图像：拖着一条“尾巴”**

为了理解为什么需要这个算符，我们可以建立一个形象的几何图像：

*   **纯费米子图像**：
    通常我们认为费米子算符 $c_j^\dagger$ 是局域的，就像在 $j$ 点放了一颗弹珠。

*   **Jordan-Wigner 费米子图像**：
    在 Jordan-Wigner 变换中，当你在位置 $j$ 产生一个费米子（对应自旋翻转 $\sigma_j^+$）时，你不仅仅是在 $j$ 点放了一个粒子，你还同时在这个粒子的左侧拖了一条 **“看不见的弦” (String)** 或者 **“尾巴”**。
    
    这条尾巴从链的最左端（$l=1$）一直延伸到当前位置（$l=j-1$）。

*   **尾巴的作用**：
    这条尾巴的作用是探测背景。任何位于这条尾巴覆盖范围内的其他粒子，都会与这条尾巴发生相互作用，贡献一个 $(-1)$ 的相位。

#### **(3) 为什么要这样设计？**

这正是为了解决“统计性质”的矛盾：

1.  **自旋（玻色子性质）**：在不同格点 $j$ 和 $k$，自旋算符是互不干扰的（对易的），即 $A B = B A$。
2.  **费米子**：在不同格点，费米子算符必须是互相排斥的（反对易的），即 $A B = - B A$。

**String 算符的巧妙之处在于：**
当你交换两个算符的位置时，比如把 $j$ 点的算符移到 $k$ 点算符的左边（设 $k < j$）：
*   费米子算符本身交换会产生一个负号（$-1$）。
*   但是，由于 $j$ 点算符拖着一条长长的 String 尾巴，这条尾巴覆盖了 $k$ 点。当 $j$ 算符跨过 $k$ 点的粒子时，String 算符会识别出 $k$ 点有粒子，从而额外产生一个负号（$-1$）。

**最终结果：**
$$ (-1)_{\text{费米子交换}} \times (-1)_{\text{String算符}} = +1 $$

两个负号相互抵消，使得这两个复合算符（自旋算符）在宏观上表现得像玻色子一样（对易），尽管它们内部是由费米子构成的。这就是 String 算符存在的根本原因。

| Oh  | E | 6C4 | 3C2 | 6C'2 | 8C3 | I  | 6IC4 | 3σh |6σd| 8IC3 |
|--------|---|-----|-----|------|-----|----|------|-----|-----|-----|
| D   | 2 | 1   | 0   | -1    | 0   | -2  |  -1  | 0   | 1   | 0   |
