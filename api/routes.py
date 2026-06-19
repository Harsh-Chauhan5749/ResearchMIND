from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from core.database import get_all_papers
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    paper_ids: Optional[List[int]] = None

class ChatRequest(BaseModel):
    session_id: str
    query: str
    paper_ids: Optional[List[int]] = None

@router.get("/papers")
def list_papers():
    return get_all_papers()

@router.post("/search")
def search_papers(req: SearchRequest):
    try:
        vs = VectorStore()
        results = vs.hybrid_search(req.query, k=req.top_k, paper_ids=req.paper_ids)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def chat(req: ChatRequest):
    try:
        pipeline = RAGPipeline()
        # Collect generator string for REST
        response_gen = pipeline.chat(req.session_id, req.query, req.paper_ids)
        full_response = ""
        for chunk in response_gen:
            full_response += chunk
        return {
            "session_id": req.session_id,
            "response": full_response,
            "sources": pipeline.last_sources,
            "confidence": pipeline.last_confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
