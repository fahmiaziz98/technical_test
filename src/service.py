import uvicorn
from fastapi import FastAPI, HTTPException
from pipeline.rag import RAGPipeline
from models import Question, Response
from pipeline.utils import init_logger

logger = init_logger()

app = FastAPI(
    title="RAG API",
    description="API for Question Answering using RAG",
    version="1.0.0"
)

rag_pipeline = RAGPipeline()


@app.post("/ask", response_model=Response)
def ask_question(question: Question):
    """
    Endpoint to ask questions to the RAG system
    """
    try:
        logger.info(f"Received question: {question.text}")
        response = rag_pipeline.get_response(question.text)
        return Response(answer=response)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
