# Context Attack v4 方法说明文档

> 代码参考：context_attack_v4.py

## 1. 方法概述

Context Attack v4 是一种面向 **上下文图像编辑模型（如 FLUX.1-Kontext）** 的对抗攻击方法，其核心目标是：

> **对齐原始参考图像（context image）和目标参考图像（reference image）之间的特征，使图像编辑模型的输出与目标参考图像相似。**

该方法基于 Context Attack v3，但移除了 context loss，专注于特征对齐，通过以下关键机制实现：

* 精确建模 FLUX 的 **latent packing 结构（target + context）**
* 基于 Transformer 内部结构的 **attention-level 对齐**
* 多层语义对齐（latent / velocity / attention / qk）
* 单独损失反向传播与梯度累积优化框架

整体优化目标为：

\[
\min ; \lambda_1 L_{\text{latent}} + \lambda_2 L_{\text{velocity}} + \lambda_3 L_{\text{attention}} + \lambda_4 L_{\text{qk}}
\]

---

## 2. 问题定义

### 2.1 输入与目标

给定：

* 输入图像（context image）：\( x \)
* 目标参考图像（reference image）：\( x_{\text{ref}} \)
* 编辑指令：\( p \)

目标：

构造对抗扰动 \( \delta \)，得到：

\[
x_{\text{adv}} = x + \delta
\]

使得：

* 模型对 \( x_{\text{adv}} \) 的编辑输出与对 \( x_{\text{ref}} \) 的编辑输出相似
* 通过对齐 latent、velocity、attention 和 QK 特征实现

---

### 2.2 攻击目标本质

Context Attack v4 的本质是：

> **最小化 adversarial 分支与 reference 分支之间的特征差异**

即：

\[
\text{minimize } \| f_{\text{adv}} - f_{\text{ref}} \|
\]

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

## 4. 多层语义对齐机制

Context Attack v4 引入 4 种对齐损失：

---

### 4.1 Latent 对齐（Ll）

\[
L_l = | z_{\text{adv}} - z_{\text{ref}} |^2
\]

作用：

* 使对抗图像 latent 接近 reference

---

### 4.2 Velocity 对齐（Lv）

\[
L_v = | \epsilon_{\text{adv}} - \epsilon_{\text{ref}} |^2
\]

作用：

* 对齐 diffusion 预测结果

---

### 4.3 Attention Map 对齐（La）

对齐：

\[
\text{softmax}(QK^T)
\]

仅在：

```
target → context attention
```

上进行：

\[
L_a = | A_{\text{adv}} - A_{\text{ref}} |^2
\]

---

### 4.4 QK 特征对齐（Lq）

\[
L_q = | Q_{\text{adv}} - Q_{\text{ref}} |^2 + | K_{\text{adv}} - K_{\text{ref}} |^2
\]

作用：

* 约束 attention 计算前的特征空间

---

## 5. 总体优化目标

最终损失函数：

\[
L = \lambda_1 L_l + \lambda_2 L_v + \lambda_3 L_a + \lambda_4 L_q
\]

其中：

| Loss  | 作用           |
| ----- | ------------ |
| \(L_l\) | latent 对齐    |
| \(L_v\) | diffusion 对齐 |
| \(L_a\) | attention 对齐 |
| \(L_q\) | 特征对齐         |

---

## 6. 梯度计算与累积

### 6.1 单独反向传播

为监督不同损失的梯度变化，每个损失单独进行反向传播：

```python
for loss_name, loss in losses.items():
    loss.backward(retain_graph=True)
    grad_norm = adv.grad.norm()
    wandb.log(f"grad/{loss_name}", grad_norm)
    total_grad += weights[loss_name] * adv.grad.clone()
    adv.grad.zero_()
```

### 6.2 梯度累积

累积加权梯度：

\[
\nabla_{\text{total}} = \sum \lambda_i \nabla_{L_i}
\]

### 6.3 梯度爆炸预防

* 每个损失反向传播后立即清零梯度
* 使用 `retain_graph=True` 允许多次反向传播
* 无需额外裁剪，依赖于 per-loss 删除

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
Ll, Lv, La, Lq = compute_losses(...)
```

---

### Step 6：单独反向传播与梯度累积

```python
total_grad = 0
for loss in [Ll, Lv, La, Lq]:
    loss.backward(retain_graph=True)
    total_grad += weight * adv.grad
    adv.grad.zero_()
adv.grad = total_grad
```

---

### Step 7：梯度更新（PGD）

```python
adv = adv + alpha * sign(grad)
```

并约束：

\[
|\delta|_\infty \le \epsilon
\]

---

### Step 8：迭代优化

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


## 9. Prompt Pool 设计与参数

支持两种模式，通过 `--prompt_mode` 参数控制：

- `multi`（默认）：使用多种编辑任务（见下）
- `single`：仅使用一个提示词 `A photo of this person`

包含 6 类编辑任务（multi模式）：

1. 表情变化
2. 配饰添加
3. 动作变化
4. 外观变化
5. 场景变化
6. 组合任务

作用：

> multi：学习"通用特征对齐能力"，而非特定任务攻击
> single：仅对齐基础人像语义，便于消融/对比实验

命令行用法：

```bash
python context_attack_v4.py ... --prompt_mode single
```

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
* 梯度监督与累积优化
* 泛化能力强（prompt pool）

---

### 局限性

* 计算开销较大（多 forward + 多 backward）
* 依赖模型结构（需要 hook）
* token slicing 依赖固定布局

---

## 12. 方法本质总结

Context Attack v4 的核心思想可以概括为：

> **通过最小化 adversarial 与 reference 分支的特征差异，实现对上下文图像的系统性"特征对齐攻击"。**</content>
<parameter name="filePath">/home/humw/Codes/ContextAttack/attack/attack_Flux_Kontext/context_attack_v4.md

