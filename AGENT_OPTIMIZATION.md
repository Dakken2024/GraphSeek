# Agent Skill 架构优化

本次重构实现了以下核心优化：

## 1. Agent 工具框架 (`tools/`)

### 核心组件
- **`tools/base.py`**: 工具抽象基类
  - `Tool`: 所有工具的基类，定义统一接口
  - `ToolResult`: 标准化的工具执行结果
  - `ToolRegistry`: 工具注册和管理中心

- **`tools/graph_tools.py`**: 图谱相关工具
  - `GraphQueryTool`: 知识图谱查询
  - `GraphStatsTool`: 图谱统计信息
  - `EntitySearchTool`: 实体搜索和邻居发现

- **`tools/retrieval_tools.py`**: 检索相关工具
  - `DocumentSearchTool`: 基础文档检索
  - `HybridSearchTool`: 混合检索（支持 reranking + GraphRAG）

- **`tools/llm_tools.py`**: LLM 相关工具
  - `TextGenerationTool`: 文本生成
  - `QueryExpansionTool`: HyDE 查询扩展

## 2. 增强 GraphRAG (`core/`)

### Agent 实现
- **`core/agent.py`**: ReAct 推理框架
  - `AgentState`: 状态管理（思考链、工具调用、观察记录）
  - `ReActAgent`: 完整的 ReAct 循环实现
    - 自动推理和行动选择
    - 多轮工具调用
    - 流式和非流式两种执行模式

### 缓存系统
- **`core/cache.py`**: 检索缓存
  - LRU 淘汰策略
  - TTL 过期机制
  - 命中率统计
  - 线程安全设计

## 3. 监控日志系统 (`utils/`)

### 性能监控
- **`utils/monitoring.py`**:
  - `Monitor`: 单例监控器
  - `MetricsCollector`: 指标收集器
  - `PerformanceTracker`: 性能追踪（装饰器/上下文管理器）
  - 支持操作耗时、错误率、吞吐量等指标

### 日志系统
- **`utils/logger.py`**:
  - 结构化 JSON 日志
  - 彩色控制台输出
  - 日志上下文管理
  - 执行时间自动记录

## 4. 服务层增强 (`services/`)

### RetrievalService 升级
- 集成检索缓存
- 性能监控埋点
- 缓存统计接口
- 可配置的 TTL 和容量

## 使用示例

### 创建 Agent 并执行查询
```python
from tools.base import ToolRegistry
from tools.graph_tools import GraphQueryTool, GraphStatsTool
from tools.retrieval_tools import HybridSearchTool
from core.agent import ReActAgent

# 初始化工具注册表
registry = ToolRegistry()
registry.register(GraphQueryTool(graph_service))
registry.register(GraphStatsTool(graph_service))
registry.register(HybridSearchTool(ensemble_retriever, reranker, graph_service))

# 创建 Agent
agent = ReActAgent(
    tool_registry=registry,
    llm_service=llm_service,
)

# 执行查询
state = agent.run("什么是量子计算？", chat_history=[])
print(state.final_answer)
```

### 使用缓存
```python
from services.retrieval_service import RetrievalService

retrieval = RetrievalService(
    ensemble_retriever=...,
    cache_enabled=True,
    cache_ttl=3600.0,  # 1 小时
)

# 自动缓存查询结果
docs = retrieval.retrieve(query="量子计算")

# 查看缓存统计
stats = retrieval.get_cache_stats()
print(f"缓存命中率：{stats['hit_rate_percent']}%")
```

### 性能监控
```python
from utils.monitoring import Monitor

monitor = Monitor()

# 方式 1: 上下文管理器
with monitor.measure("document_processing"):
    process_documents()

# 方式 2: 装饰器
@monitor.track_function("custom_operation")
def my_function():
    pass

# 获取指标
metrics = monitor.get_metrics()
```

## 架构优势

1. **模块化**: 工具、Agent、缓存、监控完全解耦
2. **可扩展**: 轻松添加新工具或替换组件
3. **可观测**: 完整的性能指标和日志追踪
4. **高性能**: LRU 缓存减少重复计算
5. **智能化**: ReAct 框架实现自主决策

## 后续建议

1. 添加更多专用工具（计算器、API 调用等）
2. 实现持久化存储（Redis/Memcached）
3. 集成 Prometheus/Grafana 监控
4. 添加单元测试覆盖
5. 实现分布式锁支持并发
