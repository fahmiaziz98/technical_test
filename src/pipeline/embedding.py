from langchain_huggingface import HuggingFaceEmbeddings
from pipeline.utils import init_logger

logger = init_logger()

def embedding_pipeline(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Initialize the embedding pipeline with the specified model name.

    Args:
        model_name (str): The name of the model to use for embedding.

    Returns:
        HuggingFaceEmbeddings: The initialized embedding pipeline.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    logger.info("Embeddings initialized successfully")
    return embeddings

