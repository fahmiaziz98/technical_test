# src/pipeline/llm.py
import os
from langchain_groq import ChatGroq
from pipeline.utils import init_logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = init_logger()

def get_llm(model: str = "llama-3.1-8b-instant"):
    """
    Initialize the LLM pipeline with the specified model name.

    Args:
        model (str): The name of the model to use for LLM.

    Returns:
        ChatGroq: The initialized LLM pipeline.
    """
    llm = ChatGroq(
        model=model,
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=3,
        streaming=True,
    )
    logger.info("LLM initialized successfully")
    return llm