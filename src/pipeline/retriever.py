# src/pipeline/retriever.py
from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from pipeline.utils import init_logger

logger = init_logger()

def get_vector_store(embeddings, collection_name, connection_string):
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )
    logger.info("Vector store initialized successfully")
    return vector_store

def get_native_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_hybrid_retriever(native_retriever, chunks, k=3, weights=[0.5, 0.5], rerank_top_n=5):
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