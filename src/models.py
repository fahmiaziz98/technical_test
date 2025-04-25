from enum import Enum
from pydantic import BaseModel
from typing import Optional, List

class RetrievalMethod(str, Enum):
    NATIVE = "native"
    HYBRID = "hybrid"

class QuestionRequest(BaseModel):
    session_id: str
    query: str
    method: RetrievalMethod

class RetrieverConfig(BaseModel):
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
    method: RetrievalMethod
    model: str
    retriever_config: RetrieverConfig

    class Config:
        from_attributes = True

class Response(BaseModel):
    session_id: str
    query: str
    answer: str
    metadata: Metadata

    class Config:
        from_attributes = True