import os
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pipeline.embedding import embedding_pipeline
from pipeline.utils import init_logger

load_dotenv(find_dotenv())

logger = init_logger()

class RAGPipeline:
    def __init__(self):
        self.connection_string = (
            f"postgresql+psycopg://"
            f"{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST')}:"
            f"{os.getenv('POSTGRES_PORT')}/"
            f"{os.getenv('POSTGRES_DB')}"
        )
        self.collection_name = os.getenv('COLLECTION_NAME')
        
        self._init_vector_store()
        self._init_llm()
        self._init_chain()
    
    def _init_vector_store(self):
        """Initialize vector store and retriever"""
        try:
            self.vector_store = PGVector(
                embeddings=embedding_pipeline(),
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def _init_llm(self):
        """Initialize LLM"""
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=3,
            streaming=True,
        )
        logger.info("LLM initialized successfully")

    def _init_chain(self):
        """Initialize RAG chain"""

        prompt = hub.pull("rlm/rag-prompt")
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized successfully")


    def get_response(self, question: str) -> str:
        """
        Get response from RAG pipeline
        
        Args:
            question (str): User question
            
        Returns:
            str: Response containing answer
        """
        try:
            logger.info(f"Processing question: {question}")
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
