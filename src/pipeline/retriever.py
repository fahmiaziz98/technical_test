# src/pipeline/retriever.py
from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from pipeline.utils import init_logger

logger = init_logger()

def get_vector_store(embeddings, collection_name, connection_string):
    """
    Initialize the vector store with the specified embeddings, collection name, and connection string.

    Args:
        embeddings (HuggingFaceEmbeddings): The embeddings to use for the vector store.
        collection_name (str): The name of the collection to use for the vector store.
        connection_string (str): The connection string to use for the vector store.

    Returns:
        PGVector: The initialized vector store.
    """
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )
    logger.info("Vector store initialized successfully")
    return vector_store

def get_native_retriever(vector_store, k=3):
    """
    Get the native retriever from the vector store.

    Args:
        vector_store (PGVector): The vector store to use for the retriever.
        k (int): The number of results to return.

    Returns:
        PGVectorRetriever: The initialized native retriever.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_hybrid_retriever(native_retriever, chunks, k=3, weights=[0.5, 0.5], rerank_top_n=5):
    """
    Get the hybrid retriever from the vector store.

    Args:
        native_retriever (PGVectorRetriever): The native retriever to use for the hybrid retriever.
        chunks (list): The list of chunks to use for the hybrid retriever.
        k (int): The number of results to return.
        weights (list): The weights to use for the hybrid retriever.
        rerank_top_n (int): The number of results to rerank.

    Returns:
        ContextualCompressionRetriever: The initialized hybrid retriever.
    """
    compressor = FlashrankRerank(top_n=rerank_top_n)
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = k
    ensemble_retriever = EnsembleRetriever(
        retrievers=[native_retriever, keyword_retriever],
        weights=weights
    )
    hybrid_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    logger.info("Hybrid retriever initialized successfully")
    return hybrid_retriever