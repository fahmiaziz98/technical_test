from langchain_huggingface import HuggingFaceEmbeddings
from pipeline.utils import init_logger

logger = init_logger()

def embedding_pipeline(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    logger.info("Embeddings initialized successfully")
    return embeddings

