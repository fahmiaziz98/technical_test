from enum import Enum
from pydantic import BaseModel
from typing import Optional, List

class RetrievalMethod(str, Enum):
    """
    Enum for the retrieval methods.
    """
    NATIVE = "native"
    HYBRID = "hybrid"

class QuestionRequest(BaseModel):
    """
    Pydantic model for the question request.
    """
    session_id: str
    query: str
    method: RetrievalMethod

class RetrieverConfig(BaseModel):
    """
    Pydantic model for the retriever config.
    """
    type: str
    collection: str
    top_k: Optional[int] = None
    vector_store_top_k: Optional[int] = None
    bm25_top_k: Optional[int] = None
    weights: Optional[List[float]] = None
    rerank_top_n: Optional[int] = None

    class Config:
        from_attributes = True

class Metadata(BaseModel):
    """
    Pydantic model for the metadata.
    """
    method: RetrievalMethod
    model: str
    retriever_config: RetrieverConfig

    class Config:
        from_attributes = True

class Response(BaseModel):
    """
    Pydantic model for the response.
    """
    session_id: str
    query: str
    answer: str
    metadata: Metadata

    class Config:
        from_attributes = True