import os
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from pipeline.embedding import embedding_pipeline
from pipeline.utils import init_logger
from models import RetrievalMethod, Metadata

load_dotenv(find_dotenv())

logger = init_logger()

class RAGPipeline:
    def __init__(self, PATH: str):
        self.connection_string = (
            f"postgresql+psycopg://"
            f"{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:"
            f"{os.getenv('POSTGRES_PORT')}/"
            f"{os.getenv('POSTGRES_DB')}"
        )
        self.collection_name = os.getenv('COLLECTION_NAME')
        self.path = PATH
        self.embeddings = embedding_pipeline()
        self.model = "llama-3.1-8b-instant"

        self.chunks = self.load_split_documents()
        self._init_vector_store()
        self._init_hybrid_retriever()
        self._init_llm()
        self._init_chains()

    def _get_retriever_config(self, method: RetrievalMethod) -> Dict[str, Any]:
        """Get retriever configuration based on method"""
        if method == RetrievalMethod.NATIVE:
            return {
                "type": "vector_store",
                "top_k": 3,
                "collection": self.collection_name
            }
        else:
            return {
                "type": "hybrid",
                "top_k": 3,
                "vector_store_top_k": 3,
                "bm25_top_k": 3,
                "weights": [0.5, 0.5],
                "rerank_top_n": 5,
                "collection": self.collection_name
            }

    def load_split_documents(self):
        """Load documents from the specified path."""
        text = UnstructuredLoader(self.path).load()
    
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )
        return text_splitter.split_documents(text)
    
    def _init_vector_store(self):
        """Initialize vector store and native retriever"""
        try:
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
            )
            self.native_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _init_hybrid_retriever(self):
        """Initialize hybrid (ensemble) retriever"""
        logger.info("Initializing hybrid retriever")
        compressor = FlashrankRerank(top_n=5)
        keyword_retriever = BM25Retriever.from_documents(self.chunks)
        keyword_retriever.k = 3

        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.native_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )   
        self.hybrid_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        logger.info("Hybrid retriever initialized successfully")

    def _init_llm(self):
        """Initialize LLM"""
        self.llm = ChatGroq(
            model=self.model,
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=3,
            streaming=True,
        )
        logger.info("LLM initialized successfully")

    def _init_chains(self):
        """Initialize both RAG chains"""
        prompt = hub.pull("rlm/rag-prompt")
        
        # Native chain
        self.native_chain = (
            {"context": self.native_retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Hybrid chain
        self.hybrid_chain = (
            {"context": self.hybrid_retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG chains initialized successfully")

    def get_response(self, session_id: str, query: str, method: RetrievalMethod) -> tuple[str, Metadata]:
        """
        Get response from RAG pipeline using specified method
        
        Args:
            session_id (str): Session identifier
            query (str): User question
            method (RetrievalMethod): Retrieval method to use
            
        Returns:
            tuple[str, Metadata]: Response containing answer and metadata
        """
        try:
            logger.info(f"Processing question for session {session_id} using {method} method")
            
            if method == RetrievalMethod.NATIVE:
                response = self.native_chain.invoke(query)
            else:  
                response = self.hybrid_chain.invoke(query)
    
            metadata = Metadata(
                method=method,
                model=self.model,
                retriever_config=self._get_retriever_config(method)
            )
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Error processing question for session {session_id}: {e}")
            raise