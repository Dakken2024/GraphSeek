"""
GraphSeek - Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History.
Developed by: Owen Valentinus &copy; All Rights Reserved 2025
2026 Enhanced: multi-backend LLM gateway, Agentic query planning, ColBERT token retrieval,
multi-objective reranking, Harness fact-checking, community summary index.
"""
import streamlit as st
from sentence_transformers import CrossEncoder

from config import AppConfig
from services.llm_service import LLMService
from services.document_service import DocumentProcessingService
from services.retrieval_service import RetrievalService
from services.colbert_retriever import ColbertRetriever
from services.harness import HarnessValidator


def initialize_app() -> tuple:
    """Initialize application configuration and services."""
    config = AppConfig.from_environment()
    
    llm_service = LLMService(
        api_url=config.models.ollama_api_url,
        model=config.models.llm_model,
    )
    
    return config, llm_service


def check_models(llm_service: LLMService, config: AppConfig) -> None:
    """Check if required models are available."""
    result = llm_service.check_models([
        config.models.llm_model,
        config.models.embeddings_model,
    ])
    
    if not result.get("available", False):
        missing = result.get("missing_models", [])
        st.error(f"⚠ Required models not found: {', '.join(missing)}")
        for model in missing:
            st.code(f"ollama pull {model}", language="bash")
        st.stop()
    
    st.sidebar.expander("📚 Model Information", expanded=False).json({
        "models": result.get("all_models", [])
    })


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "messages": [],
        "retrieval_pipeline": None,
        "rag_enabled": True,
        "documents_loaded": False,
        "enable_hyde": True,
        "enable_reranking": True,
        "enable_graph_rag": True,
        "enable_query_planning": True,
        "enable_colbert": True,
        "enable_mmr": True,
        "enable_harness": True,
        "colbert_retriever": None,
        "temperature": 0.3,
        "max_contexts": 3,
        "processing": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(config: AppConfig, llm_service: LLMService) -> None:
    """Render sidebar with document management and settings."""
    with st.sidebar:
        st.header("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        
        if uploaded_files and not st.session_state.documents_loaded:
            with st.spinner("Processing documents..."):
                try:
                    doc_service = DocumentProcessingService(
                        embedding_model=config.models.embeddings_model,
                        base_url=config.models.ollama_base_url,
                        chunk_size=config.retrieval.chunk_size,
                        chunk_overlap=config.retrieval.chunk_overlap,
                        bm25_weight=config.retrieval.bm25_weight,
                        faiss_weight=config.retrieval.faiss_weight,
                    )
                    
                    pipeline = doc_service.process_files(
                        uploaded_files,
                        reranker=st.session_state.get("reranker"),
                        llm_service=llm_service,
                        enable_communities=True,
                    )
                    
                    st.session_state.retrieval_pipeline = pipeline
                    st.session_state.documents_loaded = True
                    st.success("Documents processed!")
                    
                    graph_stats = pipeline["graph_service"].get_stats()
                    with st.expander("📊 Knowledge Graph Stats"):
                        st.write(f"🔗 Total Nodes: {graph_stats['total_nodes']}")
                        st.write(f"🔗 Total Edges: {graph_stats['total_edges']}")
                        communities = pipeline["graph_service"].community_summaries
                        st.write(f"🧩 Communities: {len(communities)}")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        st.markdown("---")
        st.header("⚙ RAG Settings")
        
        st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
        st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
        st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
        st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
        st.session_state.enable_query_planning = st.checkbox(
            "Agentic Query Planning", value=True,
            help="复杂问题自动拆分为多个子查询并行检索（替代 HyDE 单向生成）",
        )
        st.session_state.enable_colbert = st.checkbox(
            "ColBERT Token Retrieval", value=True,
            help="Token 级延迟交互检索（模型不可用时自动降级为 FAISS+BM25）",
        )
        st.session_state.enable_mmr = st.checkbox(
            "MMR Diversity", value=True,
            help="多目标重排中的信息多样性惩罚",
        )
        st.session_state.enable_harness = st.checkbox(
            "Harness Fact Check", value=True,
            help="答案生成后自动事实校验与自我修正（[UNKNOWN] 标注）",
        )
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
        
        # 多后端网关信息
        with st.expander("🤖 LLM Backend Info"):
            st.write(f"Backend: `{llm_service.backend_name}`")
            st.write(f"Model: `{llm_service.model}`")
            if llm_service.backend_name == "openai":
                st.write(f"API Base: `{config.models.llm_api_base or '默认'}`")
            st.write("切换方式: 设置环境变量 `LLM_BACKEND`/`LLM_MODEL`/`LLM_API_KEY`/`LLM_API_BASE`")
            token_stats = llm_service.get_token_stats()
            if token_stats:
                st.json(token_stats)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown(
            """
            <div style="position: absolute; top: 20px; right: 10px; font-size: 12px; color: gray;">
                <b>Developed by:</b> Owen Valentinus &copy; All Rights Reserved 2025
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_interface(llm_service: LLMService, config: AppConfig) -> None:
    """Render main chat interface."""
    st.title("🤖 GraphSeek")
    st.caption(
        "Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History"
    )
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.header("❓ Ask a Question")
    
    if prompt := st.chat_input("Ask about your documents..."):
        chat_history = "\n".join([
            msg["content"] for msg in st.session_state.messages[-5:]
        ])
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            retrieved_docs = []
            
            context = ""
            if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
                pipeline = st.session_state.retrieval_pipeline
                try:
                    # 懒加载 ColBERT 检索器（模型不可用时自动降级）
                    if st.session_state.colbert_retriever is None:
                        st.session_state.colbert_retriever = ColbertRetriever(device=config.device)
                    
                    retrieval_service = RetrievalService(
                        ensemble_retriever=pipeline["ensemble"],
                        reranker=pipeline["reranker"],
                        knowledge_graph=pipeline["knowledge_graph"],
                        graph_service=pipeline.get("graph_service"),
                        colbert_retriever=st.session_state.colbert_retriever,
                        candidate_documents=pipeline.get("documents", []),
                        llm_service=llm_service,
                    )
                    
                    docs = retrieval_service.retrieve(
                        query=prompt,
                        chat_history=chat_history,
                        enable_hyde=st.session_state.enable_hyde,
                        enable_graph_rag=st.session_state.enable_graph_rag,
                        enable_reranking=st.session_state.enable_reranking,
                        max_contexts=st.session_state.max_contexts,
                        llm_service=llm_service,
                        enable_query_planning=st.session_state.enable_query_planning,
                        enable_colbert=st.session_state.enable_colbert,
                        enable_mmr=st.session_state.enable_mmr,
                    )
                    retrieved_docs = docs
                    
                    context = "\n".join(
                        f"[Source {i+1}]: {doc.page_content}"
                        for i, doc in enumerate(docs)
                    )
                except Exception as e:
                    st.error(f"Retrieval error: {str(e)}")
            
            system_prompt = f"""Use the chat history to maintain context:
                Chat History:
                {chat_history}

                Analyze the question and context through these steps:
                1. Identify key entities and relationships
                2. Check for contradictions between sources
                3. Synthesize information from multiple contexts
                4. Formulate a structured response

                Context:
                {context}

                Question: {prompt}
                Answer:"""
            
            try:
                for token in llm_service.generate_response(
                    prompt=system_prompt,
                    temperature=st.session_state.temperature,
                    max_context=4096,
                ):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # 2026 增强：检索溯源面板 + 图谱子图卡片
                if retrieved_docs:
                    with st.expander("🔎 Retrieval Traceability", expanded=False):
                        if retrieval_service.last_rerank_details:
                            st.table(retrieval_service.last_rerank_details)
                        st.caption("多目标分: S_rel 语义 / S_graph 图中心性 / S_time 时效 / S_div 多样性")
                    _render_graph_card(pipeline.get("graph_service"), prompt)
                
                # 2026 增强：Harness 事实校验护栏
                harness_result = None
                if st.session_state.enable_harness and retrieved_docs:
                    harness = HarnessValidator(llm_service=llm_service, max_rounds=2)
                    harness_result = harness.run(full_response, retrieved_docs)
                    _render_harness_result(harness_result)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                })
                
            except Exception as e:
                st.error(f"Generation error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error.",
                })


def _render_graph_card(graph_service, query: str) -> None:
    """渲染查询相关的图谱子图卡片（matplotlib 内联）。"""
    if graph_service is None:
        return
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        related = graph_service.query_graph(query, top_k=5)
        if not related:
            return
        sub = graph_service._create_weighted_subgraph(related[:3], min_weight=0.0)
        if len(sub) == 0:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(sub, seed=42)
        nx.draw_networkx(
            sub, pos, ax=ax, with_labels=True,
            node_color="#00AAFF", node_size=700, font_size=8, font_color="black",
        )
        widths = [max(0.5, sub[u][v].get("weight", 1.0)) for u, v in sub.edges()]
        nx.draw_networkx_edges(sub, pos, ax=ax, width=widths, alpha=0.6)
        ax.set_title("Knowledge Graph Subgraph")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.caption(f"图谱卡片渲染失败: {e}")


def _render_harness_result(result) -> None:
    """渲染 Harness 事实校验结果。"""
    with st.expander("🛡️ Harness Fact Check", expanded=False):
        st.write(f"**证据支持率**: {result.support_rate:.0%}  |  **修正轮数**: {result.rounds}")
        if result.corrected:
            st.info("答案已依据证据自动修正，无证据内容以 [UNKNOWN] 标注")
        for v in result.verdicts:
            mark = "✅" if v.supported else "❌"
            st.write(f"{mark} {v.claim[:100]}")


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Owen's GraphSeek", layout="wide")
    
    custom_css = """
        <style>
            .stApp { background-color: #f4f4f9; }
            h1 { color: #00FF99; text-align: center; }
            .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
            .stChatMessage.user { background-color: #e8f0fe; }
            .stChatMessage.assistant { background-color: #d1e7dd; }
            .stButton>button { background-color: #00AAFF; color: white; }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    config, llm_service = initialize_app()
    check_models(llm_service, config)
    init_session_state()
    
    try:
        reranker = CrossEncoder(
            config.models.cross_encoder_model,
            device=config.device,
        )
        st.session_state.reranker = reranker
    except Exception as e:
        st.error(f"Failed to load CrossEncoder model: {str(e)}")
    
    render_sidebar(config, llm_service)
    render_chat_interface(llm_service, config)


if __name__ == "__main__":
    main()