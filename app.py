"""
GraphSeek - Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History.
Developed by: Owen Valentinus &copy; All Rights Reserved 2025
"""
import streamlit as st
import json
from sentence_transformers import CrossEncoder

from config import AppConfig
from services.llm_service import LLMService
from services.document_service import DocumentProcessingService
from services.retrieval_service import RetrievalService


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
                    )
                    
                    st.session_state.retrieval_pipeline = pipeline
                    st.session_state.documents_loaded = True
                    st.success("Documents processed!")
                    
                    graph_stats = pipeline["graph_service"].get_stats()
                    with st.expander("📊 Knowledge Graph Stats"):
                        st.write(f"🔗 Total Nodes: {graph_stats['total_nodes']}")
                        st.write(f"🔗 Total Edges: {graph_stats['total_edges']}")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        st.markdown("---")
        st.header("⚙ RAG Settings")
        
        st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
        st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
        st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
        st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
        
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
            
            context = ""
            if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
                try:
                    retrieval_service = RetrievalService(
                        ensemble_retriever=st.session_state.retrieval_pipeline["ensemble"],
                        reranker=st.session_state.retrieval_pipeline["reranker"],
                        knowledge_graph=st.session_state.retrieval_pipeline["knowledge_graph"],
                        graph_service=st.session_state.retrieval_pipeline.get("graph_service"),
                    )
                    
                    docs = retrieval_service.retrieve(
                        query=prompt,
                        chat_history=chat_history,
                        enable_hyde=st.session_state.enable_hyde,
                        enable_graph_rag=st.session_state.enable_graph_rag,
                        enable_reranking=st.session_state.enable_reranking,
                        max_contexts=st.session_state.max_contexts,
                        llm_service=llm_service,
                    )
                    
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