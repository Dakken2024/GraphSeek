"""
Document processing service for GraphSeek application.
Handles document loading, splitting, and indexing operations.
Enhanced with LLM-driven graph extraction and community summary index.
"""
import os
import re
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi

from services.graph_service import KnowledgeGraphService, LLMEntityExtractor


class DocumentProcessingService:
    """Service for processing and indexing documents."""
    
    def __init__(
        self,
        embedding_model: str,
        base_url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        bm25_weight: float = 0.4,
        faiss_weight: float = 0.6,
    ) -> None:
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        
        self._temp_dir = "temp"
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self) -> None:
        """Create temporary directory if it doesn't exist."""
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)
    
    def process_files(
        self,
        uploaded_files: List,
        reranker: Optional = None,
        llm_service: Optional = None,
        enable_communities: bool = True,
        graph_path: Optional[str] = None,
    ) -> dict:
        """
        Process uploaded files and create retrieval pipeline.
        
        Args:
            uploaded_files: List of uploaded file objects
            reranker: Optional reranker model for neural ranking
            llm_service: LLM 服务（用于 LLM 驱动图谱抽取与社区摘要）
            enable_communities: 是否构建社区摘要（High-level 索引）
            graph_path: 图谱持久化路径
            
        Returns:
            Dictionary containing retrieval pipeline components
        """
        documents = self._load_documents(uploaded_files)
        texts = self._split_documents(documents)
        text_contents = [doc.page_content for doc in texts]
        
        embeddings = OllamaEmbeddings(model=self.embedding_model, base_url=self.base_url)
        
        vector_store = FAISS.from_documents(texts, embeddings)
        bm25_retriever = BM25Retriever.from_texts(
            text_contents,
            bm25_impl=BM25Okapi,
            preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split(),
        )
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_store.as_retriever(search_kwargs={"k": 5}),
            ],
            weights=[self.bm25_weight, self.faiss_weight],
        )
        
        # 图谱构建：LLM 驱动抽取 + 增量 merge（无 LLM 时自动降级正则）
        llm_extractor = LLMEntityExtractor(llm_service=llm_service) if llm_service else None
        graph_service = KnowledgeGraphService(persistence_path=graph_path)
        knowledge_graph = graph_service.build_graph(texts, llm_extractor=llm_extractor)
        
        # High-level 社区摘要索引（LightRAG 双层）
        if enable_communities and len(knowledge_graph) > 0:
            graph_service.build_community_summaries(llm_service=llm_service)
        
        return {
            "ensemble": ensemble_retriever,
            "reranker": reranker,
            "texts": text_contents,
            "documents": texts,
            "knowledge_graph": knowledge_graph,
            "graph_service": graph_service,
            "llm_service": llm_service,
        }
    
    def _load_documents(self, uploaded_files: List) -> List:
        """
        Load documents from uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file in uploaded_files:
            try:
                file_path = os.path.join(self._temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                loader = self._get_loader(file_path, file.name)
                if loader:
                    documents.extend(loader.load())
                
                os.remove(file_path)
            except Exception as e:
                raise RuntimeError(f"Error processing {file.name}: {str(e)}")
        
        return documents
    
    def _get_loader(self, file_path: str, filename: str):
        """
        Get appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the file
            filename: Name of the file
            
        Returns:
            Appropriate loader instance or None
        """
        if filename.endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            return Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            return TextLoader(file_path)
        return None
    
    def _split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of text chunks
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n",
        )
        return text_splitter.split_documents(documents)
    
    def cleanup(self) -> None:
        """Clean up temporary directory."""
        if os.path.exists(self._temp_dir):
            for file in os.listdir(self._temp_dir):
                file_path = os.path.join(self._temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
