from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from app.src.utils.rag import RAGPipeline
from app.src.monitoring.feedback import FeedbackLogger
from app.src.backend.langchain_wrapper import LangChainRAG
import logging
from app.config import TOP_K

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="API for document-based question answering system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag = RAGPipeline()
feedback_logger = FeedbackLogger()
langchain_rag = LangChainRAG()

# Request/Response models
class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    use_langchain: bool = False

class AskResponse(BaseModel):
    answer: str
    sources_text: str
    sources: List[dict]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    rating: Optional[int] = None
    user_feedback: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    components: dict

# Routes
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "RAG Chatbot API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "rag_pipeline": "ok",
        "feedback_db": "ok",
        "langchain": "ok"
    }
    
    try:
        # Test feedback database
        feedback_logger.get_feedback_stats()
    except Exception as e:
        components["feedback_db"] = f"error: {str(e)}"
        logger.error(f"Feedback DB health check failed: {e}")
    
    return HealthResponse(status="healthy", components=components)

@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "ok", "message": "pong"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = feedback_logger.get_feedback_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question about the documents"""
    try:
        if req.use_langchain:
            # Use LangChain implementation
            answer = langchain_rag.query(req.question)
            response = {
                "answer": answer,
                "sources_text": "Sources not available in LangChain mode",
                "sources": []
            }
        else:
            # Use custom RAG implementation
            response = rag.answer(req.question, top_k=req.top_k)
        
        return AskResponse(**response)
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit feedback for a response"""
    try:
        rag.log_feedback(
            question=req.question,
            answer=req.answer,
            sources=req.sources,
            rating=req.rating,
            user_feedback=req.user_feedback
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/langchain/stats")
async def get_langchain_stats():
    """Get LangChain-specific statistics"""
    try:
        doc_count = langchain_rag.get_document_count()
        return {"status": "success", "document_count": doc_count}
    except Exception as e:
        logger.error(f"Error getting LangChain stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/langchain/search")
async def langchain_search(query: str, k: int = 4):
    """Perform similarity search using LangChain"""
    try:
        results = langchain_rag.similarity_search(query, k=k)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Error in LangChain search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
