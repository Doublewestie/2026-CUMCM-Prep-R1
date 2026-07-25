---
name: master-brain
description: 整合 superpowers、andrej-karpathy-skills 和 claude-science 三个顶层技能，驱动"需求澄清→架构设计→编码执行→审查交付"四步闭环工作流。适用于任何需要系统性思考的科研或工程任务，当用户提出复杂需求、启动新功能或需要结构化执行时使用。
when_to_use: >
  当用户说出"激活 master-brain"、"开始新任务"、"进入工作模式"、"启动顶层思维"、
  "来，规划一下这个任务"、"开工，帮我拆解一下"、"帮我走一遍流程"、"新任务，走标准流程"、
  "帮我梳理一下这个需求"、"按流程来"等触发短语时立即加载。
user-invocable: true
disable-model-invocation: false
compatibility: opencode
---

# Master Brain Skill（顶层思维总控制器）

<objective>
作为三个顶层思维技能的总入口，自动加载 superpowers 的"规划→编码→调试→审查"四步闭环、andrej-karpathy-skills 的"先思考、保简单、精准改动"编码哲学、以及 claude-science 的科研方法论与多智能体协作能力，将用户的模糊需求转化为结构化执行计划并驱动完整交付。
</objective>

<principles>
1. **不假设，先确认**：任何架构级决策必须经过用户确认
2. **不扩大，精准改**：每次改动只解决一个明确问题，不连带修改无关代码
3. **不遗忘，闭环验**：每次改动完成后必须运行验证，失败时有节制地修复
</principles>

<skill_loading_protocol>
激活后按顺序执行：
1. 读取 `~/.opencode/skills/superpowers/SKILL.md`（若项目级 `.opencode/skills/superpowers/` 存在则优先使用），将其核心工作流纳入本次任务
2. 读取 `andrej-karpathy-skills` 的 SKILL.md，将"先思考、保简单、精准改动"作为本次任务的编码硬约束
3. 判断任务是否含文献/图表/实验/数据分析任一要素，若是则加载 `claude-science` 对应子技能
</skill_loading_protocol>

<workflow>

<step name="需求澄清" number="1">

<action>
调用 superpowers 的 writing-plans 工作流，将用户需求分解为可执行的子任务清单。
</action>

<output>
## 📋 任务分解计划

| # | 子任务 | 优先级 | 依赖 | 预估改动量 |
|---|--------|--------|------|-----------|
| 1 | XXX    | P0     | 无   | 小/中/大  |

**完成定义（DoD）**：
- [ ] 条件 1
- [ ] 条件 2
</output>

<special_routing>
- 文献调研 → 加载 claude-science 文献技能
- 图表生成 → 加载 claude-science 图表技能
- 实验设计/数据分析 → 加载 claude-science 数据技能
</special_routing>

<persistence>
将任务分解计划 + DoD 写入当前工作目录的 `PLAN.md`。文件头部包含元数据块：

```markdown
## meta
- status: planning
- current_step: Step 1
- current_task: 0
- last_updated: YYYY-MM-DD HH:MM
```

文件尾部追加变更记录区域，每次调整均在此追加时间戳日志。
</persistence>

<constraint>
输出计划清单和 PLAN.md 后，等待用户确认再进入 Step 2（快速模式可跳过此确认）。更新 meta 中 status 为 in_progress、current_step 为 Step 2。
</constraint>

</step>

<step name="架构设计" number="2">

<action>
调用 superpowers 架构规划能力，同时应用 andrej-karpathy-skills 的"避免过度工程"原则——每个设计决策必须附带"最简方案"和"当前选择理由"两句话。
</action>

<output>
## 🏗️ 技术方案

**模块划分**：
| 模块名 | 职责 | 输入 | 输出 | 替代方案 |
|--------|------|------|------|----------|

**接口定义**：
- `function_name(param: type) -> return_type`：说明

**依赖引入**（如有新增）：
- 库名：用途 + 替代方案及不选理由
</output>

<constraint>
用户确认后方可进入 Step 3。新增依赖必须等待用户确认（即使在快速模式下）。更新 PLAN.md 中 current_step 为 Step 3。
</constraint>

</step>

<step name="编码执行" number="3">

<action>
按 Step 1 子任务顺序逐一实施。每次修改前自问：①改动是否控制在最小范围？②是否有更简单的等价实现？若答案任一为否，缩小改动范围后再动手。
</action>

<debug_workflow>
每次实现后按 superpowers 调试流程执行：
1. 编写最小复现代码
2. 隔离验证子任务核心逻辑
3. 确认通过后再集成回主干
</debug_workflow>

<failure_handling>
验证失败时：
1. 输出失败原因简要诊断
2. 尝试在不扩大改动范围的前提下就地修复
3. 上限 2 次：若自动修复 ≤2 次仍失败，停止自主修改，向用户呈现当前问题及已尝试方案，请求决策
</failure_handling>

<pause_rule>
每完成一个 P0 级子任务后自动暂停，输出"已完成 Task X，是否继续 Task Y？"等待用户确认。改动量标记为"大"的 P1 任务同样需暂停。
</pause_rule>

<self_backtrack_rule>
若在编码过程中自行发现当前架构或计划存在缺陷导致无法落地：
1. 将当前任务标记为 blocked
2. 输出偏差诊断（问题描述 + 影响范围）
3. 自动退回到 Step 2，提出增量修正方案（不重做全部，只修正受影响模块）
4. 等待用户确认后方可继续
5. 在 PLAN.md 变更记录中追加一条日志
6. 单次任务中自检回溯上限 2 次；超过则暂停并向用户请求重新协商方案
</self_backtrack_rule>

<output_format>
## 🔧 任务 X 已完成

**修改文件**：path/to/file.py
**代码变更**：
```diff
- 旧代码
+ 新代码
```
**验证结果**：✅ 通过 / ❌ 失败 + 诊断
</output_format>

</step>

<step name="审查与交付" number="4">

<action>
调用 superpowers 的 verification-before-completion 工作流，同时按 andrej-karpathy-skills 的代码简洁性标准审查每条修改：是否有过度工程？是否有可删除的冗余代码？
</action>

<output>
## 📊 最终交付报告

**修改文件清单**：
| 文件 | 操作 | 状态 |
|------|------|------|

**验证结果**：
- `{TEST_CMD}`：X passed / N/A（未找到测试命令）

**文档更新检查**：若本次改动新增接口、配置项或关键流程，须指明已更新的文档章节；若未更新，须说明原因。

**下一步建议**：
1. XXX
2. YYY（待硬件/数据到达后验证）
</output>

<test_command_discovery>
按以下顺序检测项目测试命令：
1. 读取 AGENTS.md 中声明的测试命令
2. 检查 Makefile 中的 test target
3. 根据项目类型推断（Python→pytest、Rust→cargo test、Node→npm test）
4. 以上均无 → 输出"未检测到测试命令，请手动指定"，不得静默跳过
</test_command_discovery>

<constraint>
更新 PLAN.md 中 status 为 completed、current_step 为 Step 4 (done)。
</constraint>

</step>

</workflow>

<change_request_handling>
执行过程中若用户提出变更：
1. 判断影响范围
2. 若仅影响当前未完成子任务 → 直接调整，更新 PLAN.md 对应条目
3. 若影响已完成步骤 → 退回受影响步骤，输出增量更新计划，标注与之前版本的差异点，再继续推进
</change_request_handling>

<invocation_modes>

### 正式触发（精确匹配）
- "激活 master-brain"
- "开始新任务"
- "进入工作模式"
- "启动顶层思维"
- "用 master-brain 规划 XXX"

### 口语化触发（日常开发常用）
以下说法同样有效，AI 应自动识别意图：
- "来，规划一下这个任务"
- "开工，帮我拆解一下"
- "我要开始做 XXX 了，帮我走一遍流程"
- "新任务，走标准流程"
- "帮我梳理一下这个需求"
- "准备搞一个 XXX，按流程来"

### 模式选择
| 模式 | 触发方式 | 行为 |
|------|----------|------|
| 安全模式（默认） | 不加额外关键字 | 每一步架构决策都等待用户确认 |
| 快速模式 | 指令中带"自动执行"或"直接走" | 跳过 Step 1/2 的逐项确认，但新增依赖仍强制确认 |

</invocation_modes>

<auto_trigger_rules>

### 自动匹配触发（高复杂度场景）
当请求满足以下**任意 2 条**时，AI 应主动询问是否启动：
1. 涉及多文件修改或新增模块
2. 需要架构设计或技术选型
3. 用户消息长度超过 200 字
4. 包含"方案""设计""架构""重构"中任一关键词

**模糊时协商**：若 AI 不确定，必须用如下问句主动确认，不得静默决策：
> "这个任务有一定复杂度，需要我启动 master-brain 做一次完整规划吗？"

### 排除规则（不触发）
- 单函数/单行代码补全
- 错误信息解读
- 配置参数查询
- 用户明确说"快速"或"不用规划"

</auto_trigger_rules>

<decision_boundary>

| 类型 | 权限 |
|------|------|
| 模块拆分、接口设计、新增依赖 | 必须询问用户，等待确认 |
| 变量命名、代码风格、错误处理 | AI 自行判断，注释中说明 |
| 测试用例新增/修改 | AI 自行判断，Step 4 中说明覆盖范围 |
| 接口/配置变化 | 必须更新文档并在 Step 4 中说明 |
| AI 自检回溯（改架构/计划） | 必须等待用户确认 |

</decision_boundary>

<examples>

**正式触发**：
> "激活 master-brain，帮我修复 watchdog 未接入的问题。"

**口语化触发**：
> "来，规划一下这个任务，我要修 watchdog。"

**自动执行模式**：
> "开始新任务，自动执行，帮我实现 IMU 模拟器。"

**继续上次任务**：
> "继续执行 PLAN.md，从 Task 3 开始。"

**中途变更**：
> "等一下，IMU 模拟器不用输出到 COM 口，先只打印到终端。"

**预期执行流程**：
1. Step 1 → 输出任务分解 → 生成 PLAN.md（含 meta 头）
2. 确认 → Step 2 → 输出技术方案
3. 确认 → Step 3 → 逐条执行 + P0 任务暂停确认 + 失败按协议处理
4. Step 4 → 输出交付报告 + 文档检查 + 下一步建议

</examples>

<context>
相关技能参考：
- superpowers：`~/.opencode/skills/superpowers/SKILL.md`
- andrej-karpathy-skills：`~/.opencode/skills/andrej-karpathy-skills/SKILL.md`
- claude-science：`~/.opencode/skills/claude-science/SKILL.md`
</context>
