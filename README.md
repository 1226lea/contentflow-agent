# ContentFlow Agent

ContentFlow 是我的硕士毕业项目，一个创作者推荐与协作平台，使用 TF-IDF 文本相似度推荐创作方向。本项目是它的 **AI Agent 升级版**：用户用自然语言（包括中文）描述自己，大模型自主决定调用哪些工具，推荐创作方向、匹配合作需求，并起草合作申请。

## 为什么要升级

原来的 TF-IDF 推荐有三个局限：

| 局限 | 例子 | Agent 的做法 |
|---|---|---|
| 只能精确匹配词语 | 用户输入「旅行 摄影」，英文数据里一个结果也匹配不到 | 模型先理解用户需求，再把它转写成英文关键词去调用工具 |
| 需要填写结构化表单 | 用户必须分别填写兴趣、技能、目标 | 用户直接聊天；信息不够时，模型会追问 |
| 推荐和合作是两个独立功能 | 用户要自己在两个页面之间来回操作 | 模型把推荐方向、找合作、起草申请串成一个流程 |

## 架构

```mermaid
flowchart LR
    U[用户] --> L[Agent 循环<br/>agent/core.py]
    L -- 对话 + 工具定义 --> M[大模型<br/>DeepSeek / Claude]
    M -- tool_calls --> L
    L --> T1[recommend_strategies<br/>原 TF-IDF 推荐]
    L --> T2[search_collaborations]
    L --> T3[get_details]
    L --> T4[draft_application<br/>只生成草稿]
    T4 --> C{用户确认?}
    C -- 是 --> S[提交]
    L -. 模型故障 / 超出步数 .-> F[降级到 TF-IDF]
```

## 设计要点

- **Agent 循环自己实现**：用的是 LangChain 的 `bind_tools`，但没有直接调用现成的 Agent 执行器，而是自己写了一个“模型 → 工具 → 模型”的循环，每一步都看得见、可控制。
- **复用原有算法**：原来的 TF-IDF 推荐被封装成工具 `recommend_strategies`，维度和权重与原项目一致（兴趣 0.35、技能 0.30、目标 0.20、条件 0.15）。新旧方案是结合关系，不是替代关系。
- **可解释**：工具会返回匹配到的关键词和分数。系统提示词要求模型只推荐工具返回的结果，并说明推荐理由，避免模型凭空编造。
- **稳定性**：
  - 设置最大步数，防止死循环；
  - 工具报错时，把错误信息返回给模型，让它自己修正；
  - 模型调用失败时，自动降级到 TF-IDF 推荐。
- **人在回路**：`draft_application` 只生成申请草稿，由用户确认后才会提交。
- **模型可切换**：在 `.env` 里改一行 `LLM_PROVIDER`，就能在 DeepSeek 和 Claude 之间切换，方便做对比评估。

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # 然后在 .env 里填写自己的 API Key
.venv/bin/python cli.py     # 或者用 cli.py claude 指定模型
```

测试不需要 API Key：用一个按剧本回复的假模型，验证工具调用、错误处理、降级和草稿确认这几个流程。

```bash
.venv/bin/python -m pytest -q
```

## 数据

`data/demo_data.json` 是从原 ContentFlow 项目的数据迁移和演示数据中只读导出的，包括 12 个内容方向、7 个合作需求和 4 位创作者。原项目没有做任何修改。

## 下一步计划

- [ ] 评估体系：构建测试集，对比 TF-IDF、向量检索和 Agent 三种方案（命中率、延迟、成本），以及 DeepSeek 与 Claude 两个模型
- [ ] 向量检索：用 PostgreSQL + pgvector 实现语义检索，取代关键词匹配
- [ ] Web 页面：接入 Django，支持流式输出
