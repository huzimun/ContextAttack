下面给出基于你提供代码（Context Attack v3）的**完整说明文档（Markdown格式）**，采用学术风格组织，涵盖方法动机、核心机制、实现细节与工程设计。

---

# Context Attack v3 方法说明文档

> 代码参考：

## 1. 方法概述

Context Attack v3 是一种面向 **上下文图像编辑模型（如 FLUX.1-Kontext）** 的对抗攻击方法，其核心目标是：

> **破坏模型对输入上下文图像（context image）的语义依赖能力，使模型在编辑任务中无法正确利用上下文信息。**

该方法继承并扩展了 DeContext 攻击框架，通过以下关键机制实现更强攻击能力：

* 精确建模 FLUX 的 **latent packing 结构（target + context）**
* 基于 Transformer 内部结构的 **attention-level 控制**
* 多层语义对齐（latent / velocity / attention / qk）
* 统一的多目标优化框架

整体优化目标为：

[
\max ; L_{\text{context}} - \lambda_1 L_{\text{latent}} - \lambda_2 L_{\text{velocity}} - \lambda_3 L_{\text{attention}} - \lambda_4 L_{\text{qk}}
]

---

## 2. 问题定义

### 2.1 输入与目标

给定：

* 输入图像（context image）：( x )
* 参考图像（reference image）：( x_{\text{ref}} )
* 编辑指令：( p )

目标：

构造对抗扰动 ( \delta )，得到：

[
x_{\text{adv}} = x + \delta
]

使得：

* 模型生成结果不再依赖原始 context 信息
* 同时语义结构向 reference 分支对齐

---

### 2.2 攻击目标本质

Context Attack 的本质是：

> **降低 target token 对 context token 的 attention 分配**

即：

[
\text{minimize } \text{Attention}(target \rightarrow context)
]

---

## 3. FLUX 架构建模

### 3.1 Token 结构

在 FLUX Kontext 模型中，Transformer 输入 token 结构为：

#### Single-stream block：

```
[text tokens | target tokens | context tokens]
```

对应代码中的划分：

* text: `[0, 512)`
* target: `[512, 1536)`
* context: `[1536, 2560)`

#### Double-stream block：

```
[target tokens | context tokens]
```

---

### 3.2 Latent Packing

FLUX 使用 packed latent 表示：

```python
packed_input = [target_latents | context_latents]
```

其中：

* target latent：由随机噪声生成
* context latent：由输入图像编码

---

## 4. 核心机制：Context Proportion

### 4.1 定义

Context proportion 衡量：

> target token 在 attention 中分配给 context token 的比例

计算流程：

1. 获取 Q, K
2. 加入 RoPE
3. 计算 attention：

[
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)
]

4. 统计：

[
\text{ContextProp} = \mathbb{E} \left[ A_{target \rightarrow context} \right]
]

---

### 4.2 Context Loss

定义为：

[
L_c = 1 - \text{ContextProp}
]

作用：

* **最大化 ( L_c )** → 减少 context 使用

---

## 5. 多层语义对齐机制

Context Attack v3 引入 4 种对齐损失：

---

### 5.1 Latent 对齐（Ll）

[
L_l = | z_{\text{adv}} - z_{\text{ref}} |^2
]

作用：

* 使对抗图像 latent 接近 reference

---

### 5.2 Velocity 对齐（Lv）

[
L_v = | \epsilon_{\text{adv}} - \epsilon_{\text{ref}} |^2
]

作用：

* 对齐 diffusion 预测结果

---

### 5.3 Attention Map 对齐（La）

对齐：

[
\text{softmax}(QK^T)
]

仅在：

```
target → context attention
```

上进行：

[
L_a = | A_{\text{adv}} - A_{\text{ref}} |^2
]

---

### 5.4 QK 特征对齐（Lq）

[
L_q = | Q_{\text{adv}} - Q_{\text{ref}} |^2 + | K_{\text{adv}} - K_{\text{ref}} |^2
]

作用：

* 约束 attention 计算前的特征空间

---

## 6. 总体优化目标

最终损失函数：

[
L = w_c L_c - w_l L_l - w_v L_v - w_a L_a - w_q L_q
]

其中：

| Loss  | 作用           |
| ----- | ------------ |
| (L_c) | 破坏 context   |
| (L_l) | latent 对齐    |
| (L_v) | diffusion 对齐 |
| (L_a) | attention 对齐 |
| (L_q) | 特征对齐         |

---

## 7. 攻击流程

### Step 1：初始化

* 输入图像 → adv
* reference 图像 → ref
* 编码为 latent

---

### Step 2：采样 prompt

从 prompt pool 中随机选择：

```python
prompt = random.choice(prompt_pool)
```

作用：

* 提高泛化能力
* 避免 overfitting

---

### Step 3：构建输入

```python
packed_input = [noisy_latent | cond_latent]
```

---

### Step 4：Transformer 前向传播

分别计算：

* adv 分支
* ref 分支

并通过 hook 获取：

* Q / K 特征

---

### Step 5：计算损失

调用：

```python
compute_losses(...)
```

---

### Step 6：梯度更新（PGD）

```python
adv = adv + alpha * sign(grad)
```

并约束：

[
|\delta|_\infty \le \epsilon
]

---

### Step 7：迭代优化

重复上述过程：

```python
for step in attack_steps:
```

---

## 8. Attention Hook 机制

通过 hook 捕获：

```python
block.attn.norm_q
block.attn.norm_k
```

实现：

* Q/K 提取
* attention 重建
* 多层监督

---

## 9. Prompt Pool 设计

包含 6 类编辑任务：

1. 表情变化
2. 配饰添加
3. 动作变化
4. 外观变化
5. 场景变化
6. 组合任务

作用：

> 学习“通用上下文破坏能力”，而非特定任务攻击

---

## 10. 工程优化设计

### 10.1 Reference 缓存

```python
packed_cond_ref (cached)
```

避免重复计算

---

### 10.2 Prompt 预编码

```python
Pre-encoded 59 prompts
```

减少文本编码开销

---

### 10.3 Mixed Precision

支持：

* fp16
* bf16

---

### 10.4 Hook 管理

避免：

* memory leak
* hook 覆盖 bug

---

## 11. 方法特点总结

### 优点

* 精确控制 Transformer 内部机制
* 多层语义约束（强表达能力）
* 适配 FLUX Kontext 架构
* 泛化能力强（prompt pool）

---

### 局限性

* 计算开销较大（多 forward）
* 依赖模型结构（需要 hook）
* token slicing 依赖固定布局

---

## 12. 方法本质总结

Context Attack v3 的核心思想可以概括为：

> **通过削弱 target 对 context 的 attention，同时将语义对齐至 reference 分支，从而实现对上下文图像的系统性“失效攻击”。**
