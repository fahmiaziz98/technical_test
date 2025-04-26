import os
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from pipeline.embedding import embedding_pipeline
from pipeline.llm import get_llm
from pipeline.retriever import get_vector_store, get_native_retriever, get_hybrid_retriever
from pipeline.utils import init_logger
from models import RetrievalMethod, Metadata

load_dotenv(find_dotenv())

logger = init_logger()

class RAGPipeline:
    def __init__(self, PATH: str):
        """
        Initialize the RAG pipeline with the specified path.

        Args:
            PATH (str): The path to the documents to be used for keyword search.
        """
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

        self.chunks = self.load_split_documents(self.path)
        self.vector_store = get_vector_store(self.embeddings, self.collection_name, self.connection_string)
        self.native_retriever = get_native_retriever(self.vector_store)
        self.hybrid_retriever = get_hybrid_retriever(self.native_retriever, self.chunks)
        self.llm = get_llm(self.model)
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

    def load_split_documents(self, path: str):
        """
        Load documents from the specified path and split them into chunks.

        Args:
            path (str): The path to the documents to be used for keyword search.

        Returns:
            list: A list of documents.
        """
        text = UnstructuredLoader(path).load()
    
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

    def _init_chains(self):
        """
        Initialize both RAG chains.

        Returns:
            None
        """
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
            valid_methods = {RetrievalMethod.NATIVE, RetrievalMethod.HYBRID}
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid retrieval method: {method}. "
                    f"Valid methods are: {[m.name for m in valid_methods]}"
                )

            logger.info(f"Processing question for session {session_id} using {method} method")
            
            if method == RetrievalMethod.NATIVE:
                response = self.native_chain.invoke(query)
            elif method == RetrievalMethod.HYBRID:  
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