---
name: commit-conventions
description: "全局 commit 规范：全中文格式 (type(scope): 中文描述 + 中文 body)。专有名词白名单、scope 模板、反模式。项目级细节由 .opencode/COMMIT.md 方言覆盖。Triggers: commit规范, 提交规范, commit conventions, 怎么写commit."
metadata:
  version: "1.0.0"
  last_updated: "2026-07-10"
  status: active
  task_type: workflow
---

# 全局 Commit 规范

适用于所有项目。项目级细节由各项目 `.opencode/COMMIT.md` 方言覆盖。

---

## 0. Principles

| 原则 | 含义 |
|------|------|
| Atomic | 一个 commit = 一个逻辑变更 |
| Leaves repo green | 每 commit 后测试全过 |
| Why over what | body 写决策理由，不写 diff 能看出的东西 |
| Imperative | 中文动词（添加/修复/重构），英文 subject 中保留 imperative |
| Searchable | scope/body 含模块名、函数名 |

---

## 1. 格式

```
type(scope): 中文描述

- 中文 bullet point 1
- 中文 bullet point 2
```

规则：
- `type(scope):` 保留英文小写（Conventional Commits 标准）
- 冒号后全部中文（subject description + body）
- body 不超 72 字/行
- 专有名词保持原文（见 §2）
- 不写 `##` 标题、不写 `Co-authored-by`、不用 emoji

示例：
```
feat(encoders/mcpe): 为 MCPEv2 添加 per-μ LIF 时间积分

- 将 16 个独立 LIF 拆分为 4 组 multi-output Leaky，按 μ 共享实例
- 参考 Pedersen Fig 9：空间特征被所有时间通道共享
- 测试：test_mcpe_v2.py 5/5 ✅
```

## 2. 专有名词白名单（禁止翻译）

以下术语在 commit message 中保持原文，不可用中文替代：

| 类别 | 术语 |
|------|------|
| 框架/库 | `PyTorch`, `snnTorch`, `NumPy`, `SciPy`, `CUDA` |
| ML 术语 | `LIF`, `SNN`, `CNN`, `RNN`, `BPTT`, `TDD`, `MEI`, `ZOH` |
| 算法/方法 | `Softplus`, `ReLU`, `Adam`, `SGD`, `CrossEntropyLoss` |
| 数据集 | `CPLID`, `TTPLA`, `Vergara`, `MNIST`, `CIFAR` |
| 论文作者 | `Pedersen`, `Xue`, `Hussaini`, `Pan` |

项目级额外专有名词在各项目 `.opencode/COMMIT.md` §1 中定义。

## 3. Types

| Type | 使用场景 | 中文说明 |
|------|------|------|
| `feat` | 新模块/新功能 | 添加 xx 功能 |
| `fix` | bug 修复 | 修复 xx 问题 |
| `perf` | 速度优化 | 优化 xx 性能 |
| `refactor` | 重构 | 重构 xx 模块 |
| `test` | 测试 | 添加/修改 xx 测试 |
| `docs` | 文档 | 更新/添加 xx 文档 |
| `chore` | 杂项 | 构建/依赖/配置更新 |
| `review` | 审查 | 自我审查记录 |

## 4. Anti-Patterns

- [x] 专有名词被翻译：`PyTorch` → `火炬`、`LIF` → `漏积分`
- [x] 混合 commit：feat + fix + docs 混杂
- [x] 过度简化：`更新代码` → 需写具体改了什么
- [x] commit 后不跑测试
- [x] emoji 出现在 commit message 中
