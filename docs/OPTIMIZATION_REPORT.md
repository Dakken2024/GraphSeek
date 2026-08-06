# GraphSeek 2026 系统优化升级报告

**日期**：2026-08-06
**依据文档**：[GraphSeek.md](../GraphSeek.md)（2026+ 技术迭代与架构演进白皮书 v2.0）
**目标**：将 GraphSeek 从"经典 RAG 组合"向"Agentic GraphRAG 智能体系统"演进落地
**约束**：Python 3.12+ 兼容；所有增强"可选启用 + 自动降级"，不破坏既有功能路径

---

## 一、差距分析（现状 vs 白皮书要求）

| 维度 | 优化前现状 | 白皮书目标 | 本次完成度 |
| :--- | :--- | :--- | :--- |
| 模型接入 | 仅 Ollama `/api/generate`（DeepSeek-7B） | Qwen3.x-27B + vLLM + 云 API 可切换 | ✅ 多后端网关（Ollama/OpenAI 兼容/mock） |
| 结构化输出 | 无（文本生成 + 正则解析） | Pydantic JSON Schema 强制 | ✅ `generate_structured` 自动重试校验 |
| 图谱构建 | 正则实体抽取、全局静态图 | LLM 驱动 NER/RE + 增量 merge（Extractor+Merger） | ✅ LLM 抽取器 + `merge_subgraph` 消歧合并 |
| 图谱索引 | 单层实体图 | LightRAG 双层（Low-level 实体 + High-level 社区摘要） | ✅ Louvain 社区检测 + LLM 摘要 + `query_community` |
| 检索 | FAISS + BM25 加权融合 | + ColBERT/SPLADE Token 级第三路 + RRF | ✅ ColBERT 轻量接入 + RRF 融合（基座保留降级） |
| 重排 | CrossEncoder 单一语义分 | 多目标（语义/图中心性/时效/MMR 多样性） | ✅ `MultiObjectiveReranker` 四目标加权 |
| 查询规划 | HyDE | Agentic 查询分解（Plan-and-Solve） | ✅ `QueryPlanner` 结构化分解，HyDE 保留可选 |
| 事实护栏 | 无 | Harness Loop 校验 + 自修正 | ✅ `HarnessValidator`（LLM/启发式双路径） |
| 上下文压缩 | 无 | LLMLingua-2 提纯 | ✅ `ContextCompressor`（LLM/启发式双路径） |
| 评估体系 | 无测试、无 RAGAS | RAGAS/DeepEval CI/CD | ✅ 自研 RAGAS 风格评估器 + CLI + 报告 |
| 工程测试 | 无 | 自动化评估 | ✅ 58 项 pytest 单测 + 端到端冒烟 |
| UI | 简单 Streamlit | Django6+Vue3+shadcn（另立项） | ✅ Streamlit 增强（溯源/图谱卡/后端切换/护栏展示） |

---

## 二、优化内容与实施步骤

### 阶段 1 · LLM 多后端网关（基座换擎）
**新增文件**：[services/llm_gateway.py](../services/llm_gateway.py)
- `LLMBackend` 抽象基类 + 三个实现：
  - `OllamaBackend`：原生 `/api/chat`（兼容现有 deepseek 部署）
  - `OpenAICompatBackend`：OpenAI SDK，覆盖 vLLM / SGLang / 硅基流动 / ModelScope
  - `MockBackend`：离线开发与测试
- `LLMGateway`：统一入口，支持流式/非流式/chat、指数退避重试、Token 统计、性能监控
- **Pydantic 结构化输出**：`generate_structured(schema)` 通过 `response_format`（json_schema→json_object→纯 Prompt 三级降级）+ 解析失败自动携带校验错误重试，**彻底消灭正则解析**
- **配置切换**：`LLM_BACKEND=ollama|openai|mock`、`LLM_MODEL`、`LLM_API_KEY`、`LLM_API_BASE`（环境变量）

**重构文件**：[services/llm_service.py](../services/llm_service.py) — 保持公开 API 不变的网关薄封装；[config.py](../config.py) 增加后端配置；[docker-compose.yaml](../docker-compose.yaml) 增加网关环境变量示例。

### 阶段 2 · 图谱增强（LightRAG 双层 + LLM 驱动抽取）
**修改文件**：[services/graph_service.py](../services/graph_service.py)
- Pydantic Schema：`Entity` / `Relationship` / `SubGraph` / `CommunitySummary`
- `LLMEntityExtractor`：网关结构化输出抽取 NER/RE（温度 0.1），**失败自动降级正则**（`EnhancedEntityExtractor`）
- `merge_subgraph`（Merger Agent）：归一化实体消歧（大小写/空白/连字符不敏感）、属性合并、边权重累加，支持**增量建图**
- `add_documents_incremental`：按文档抽取子图并 merge，无需全局重建（图随需而建）
- `detect_communities`（Louvain）→ `build_community_summaries` → **High-level 社区摘要图**（LLM 摘要，无 LLM 用规则摘要）
- 双检索入口：`query_graph`（Low-level PageRank/BFS）+ `query_community`（High-level 宏观）

**修改文件**：[services/document_service.py](../services/document_service.py) — `process_files` 接入 LLM 抽取器与社区摘要，管道返回分块 Document。

### 阶段 3 · 检索链路升级（演进 2/3/4）
**新增文件**：
- [services/colbert_retriever.py](../services/colbert_retriever.py)：sentence-transformers 轻量加载 `colbert-ir/colbertv2.0`，Query×Doc Token 延迟交互 MaxSim 打分；**懒加载 + 失败自动降级**（不依赖 colbert-ai/ragatouille 重型库）
- [services/reranker.py](../services/reranker.py)：`MultiObjectiveReranker` 综合评分 `Score = α·S_rel + β·S_graph + γ·S_div + δ·S_time`；`S_graph` 用图中心性、`S_div` 用 MMR 增量选择、`S_time` 用指数衰减；缺失信号取中性值可降级
- [services/query_planner.py](../services/query_planner.py)：`QueryPlanner` 结构化分解复杂查询（≤3 子查询），启发式简单问题直接检索，LLM 失败回退单查询

**修改文件**：[services/retrieval_service.py](../services/retrieval_service.py) — 三路召回（基座 FAISS+BM25 **始终保留** / ColBERT / 图谱双层）+ **RRF 融合** + 跨子查询去重 + 多目标重排，`last_rerank_details` 供溯源面板；[tools/retrieval_tools.py](../tools/retrieval_tools.py) 修复 langchain 0.3 兼容。

### 阶段 4 · Harness 护栏与上下文压缩
**新增文件**：
- [services/harness.py](../services/harness.py)：`HarnessValidator` 三步回路——断言提取（LLM/启发式）→ 证据校验（LLM-as-a-Judge）→ 自修正（≤2 轮，无证据内容以 `[UNKNOWN]` 标注），输出支持率
- [services/context_compressor.py](../services/context_compressor.py)：LLM 结构化关键句抽取 / 启发式查询词保留双路径，控制 TTFT

### 阶段 5 · 评估与测试
**新增目录**：
- [evaluation/](../evaluation/)：`RAGEvaluator`（Faithfulness / Context Relevance / Answer Relevance 三大 RAGAS 核心指标，LLM/启发式双路径）、Markdown/JSON 报告、CLI（`python -m evaluation.run_evaluation`，支持 ollama/openai/mock 三后端）、示例数据
- [tests/](../tests/)：6 个测试文件 + 端到端冒烟，**58 项全部通过**（详见第四节）

**新增工具**：[utils/text_utils.py](../utils/text_utils.py) — 中文二元组分词 + 全角/半角句子切分，解决 Python `re` 将连续中文切为单 token 导致的相似度失真。

### 阶段 6 · Streamlit 增强
**修改文件**：[app.py](../app.py)
- **检索溯源面板**：展示每次检索的多目标分（S_rel/S_graph/S_time/S_div/final）
- **图谱子图卡片**：matplotlib 内联渲染查询相关子图
- **Harness 展示**：答案生成后自动校验，展示证据支持率与逐断言 ✅/❌
- **后端信息与切换指引** + Token 统计
- 新开关：Agentic Query Planning / ColBERT / MMR Diversity / Harness Fact Check

---

## 三、白皮书对齐矩阵

| 白皮书章节 | 落地位置 |
| :--- | :--- |
| 演进 1 LightRAG + Agentic 动态子图 | `graph_service.py`（LLMEntityExtractor / merge_subgraph / 社区摘要） |
| 演进 2 ColBERT Token 级多路召回 | `colbert_retriever.py` + `retrieval_service.py` RRF |
| 演进 3 多目标重排序 | `reranker.py`（S_graph/S_time/S_div） |
| 演进 4 Agentic Query Planning | `query_planner.py`（替代 HyDE，HyDE 保留可选） |
| 附录 A 三智能体协作流 | Extractor=`LLMEntityExtractor`、Merger=`merge_subgraph`、Router≈`QueryPlanner`+`RetrievalService` 路由 |
| 附录 B 分层路由与协同公式 | `query_graph`（微观）+ `query_community`（宏观）+ `MultiObjectiveReranker` |
| Harness Loop 验证护栏 | `harness.py` |
| 上下文压缩 | `context_compressor.py` |
| 原生结构化输出 | `LLMGateway.generate_structured`（Pydantic） |
| RAGAS/DeepEval 评估流水线 | `evaluation/`（自研对齐 RAGAS 定义，可集成 CI） |

---

## 四、测试验证结果

### 4.1 单元测试（pytest）

| 测试文件 | 覆盖内容 | 结果 |
| :--- | :--- | :--- |
| tests/test_llm_gateway.py | JSON 提取、结构化输出重试、后端配置、token 统计 | ✅ 13 passed |
| tests/test_graph_service.py | 增量合并、实体消歧、边权重累加、社区摘要、持久化往返 | ✅ 10 passed |
| tests/test_retrieval.py | RRF、多目标重排（图中心性/MMR/时效）、查询规划、RetrievalService 降级 | ✅ 14 passed |
| tests/test_harness.py | 断言提取、证据校验、自修正回路、[UNKNOWN] 标注 | ✅ 8 passed |
| tests/test_compressor.py | 启发式压缩、句子切分 | ✅ 4 passed |
| tests/test_evaluation.py | 三大指标、批量评估、报告生成 | ✅ 8 passed |
| tests/test_smoke_e2e.py | **端到端冒烟**：规划→召回→RRF→重排→生成→Harness→评估→压缩 | ✅ 1 passed |
| **合计** | | **58 passed, 0 failed**（26.5s） |

### 4.2 端到端验证（mock 网关，离线）
- 完整 2026 链路（查询分解 → 三路召回 → RRF → 多目标重排 → 生成 → Harness 校验 → RAGAS 评估 → 上下文压缩）跑通
- **ColBERT 模型不可用 → 自动降级为 FAISS+BM25 双路**，检索功能不中断（已验证）
- **无 LLM 环境 → 图谱正则抽取、社区规则摘要、Harness/评估启发式降级**（已验证）

### 4.3 评估 CLI 演示（mock 后端）
```bash
python -m evaluation.run_evaluation --backend mock --output evaluation/demo_report.md
```
输出摘要：`faithfulness_avg=1.0, context_relevance_avg=0.4444, answer_relevance_avg=0.85`（mock 确定性响应，验证 LLM 评估路径；真实模型/数据时以实际为准）

### 4.4 环境与兼容性
- Python 3.12.7 + Anaconda；langchain classic 0.3 线（与项目现有 API 兼容，requirements.txt 已固定版本，勿升级 1.x）
- 全部模块 `py_compile` 通过；`import app` 通过（编辑器诊断 0 错误）

---

## 五、降级与兼容性设计（白皮书"黄金底座"保留）

| 场景 | 行为 |
| :--- | :--- |
| ColBERT 模型加载失败/离线 | 自动降级 FAISS+BM25 双路（白皮书：保留基础召回） |
| LLM 后端不可用 | 图谱正则抽取、社区规则摘要、Harness/评估启发式判定 |
| 结构化输出多次失败 | 三级 response_format 降级 + 重试携带校验错误 |
| 查询规划失败 | 回退单查询（HyDE 可独立开关） |
| 简单问题 | 不做分解，直接检索（零开销） |

---

## 六、后续建议（Roadmap 剩余项）

1. **实际部署 Qwen3.x-27B**：vLLM/SGLang + AWQ 4-bit，配置 `LLM_BACKEND=openai` + `LLM_API_BASE` 即可接入（硅基流动 / ModelScope / 自建 vLLM）
2. **Django 6 + Vue 3 + shadcn-vue 全栈重构**（白皮书附录 D）：本此保持 Streamlit，建议独立项目立项
3. **官方 ragas 集成**：当前自研评估器对齐 RAGAS 定义，需要更全面指标时可替换
4. **生产化**：Celery/Redis 后台建图任务、多租户、Prometheus 指标导出
5. **CI/CD 集成**：`pytest tests` 已可接入流水线；评估 CLI 可作 nightly 质量门禁

---

*报告生成：2026-08-06，本次优化共新增 10 个源文件、重构 8 个文件，58 项测试全绿。*
