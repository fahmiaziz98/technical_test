import os
import uvicorn
from fastapi import FastAPI, HTTPException, APIRouter
from pipeline.rag import RAGPipeline
from models import QuestionRequest, Response
from pipeline.utils import init_logger

logger = init_logger()

app = FastAPI(
    title="RAG API",
    description="API for Question Answering using RAG with multiple retrieval methods",
    version="1.0.0"
)

router = APIRouter(prefix="/api/v1")

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
rag_pipeline = RAGPipeline(os.path.join(PROJECT_DIR, "output", "combined_output.txt"))

@router.post("/ask", response_model=Response)
async def ask_question(request: QuestionRequest):
    """
    Endpoint to ask questions to the RAG system
    """
    try:
        logger.info(f"Received question for session {request.session_id} using {request.method}")
        
        answer, metadata = rag_pipeline.get_response(
            session_id=request.session_id,
            query=request.query,
            method=request.method
        )
        
        response_data = {
            "session_id": request.session_id,
            "query": request.query,
            "answer": answer,
            "metadata": {
                "method": metadata.method,
                "model": metadata.model,
                "retriever_config": {
                    "type": metadata.retriever_config.type,
                    "collection": metadata.retriever_config.collection,
                    "top_k": metadata.retriever_config.top_k,
                    "vector_store_top_k": metadata.retriever_config.vector_store_top_k,
                    "bm25_top_k": metadata.retriever_config.bm25_top_k,
                    "weights": metadata.retriever_config.weights,
                    "rerank_top_n": metadata.retriever_config.rerank_top_n
                }
            }
        }
        
        return Response(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing request for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8000)