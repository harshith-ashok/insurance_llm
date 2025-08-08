from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
from document_processor import DocumentProcessor
from semantic_search import SemanticSearch
from decision_engine import DecisionEngine
import logging

app = FastAPI(title="HackRX LLM Query System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


class QueryResponse(BaseModel):
    answers: List[Dict[str, Any]]


async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid authorization header")

    token = authorization.replace("Bearer ", "")
    expected_token = os.getenv("BEARER_TOKEN", "default_token")

    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")

    return token


@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    try:
        logger.info(
            f"Processing query with {len(request.questions)} questions")

        document_processor = DocumentProcessor()
        semantic_search = SemanticSearch()
        decision_engine = DecisionEngine()

        documents_content = await document_processor.process_document(request.documents)

        answers = []
        for question in request.questions:
            relevant_clauses = await semantic_search.search_clauses(question, documents_content)
            decision_result = await decision_engine.evaluate_decision(question, relevant_clauses)

            answers.append({
                "question": question,
                "answer": decision_result["answer"],
                "rationale": decision_result["rationale"],
                "relevant_clauses": decision_result["relevant_clauses"],
                "confidence": decision_result["confidence"]
            })

        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "hackrx-llm-system"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
