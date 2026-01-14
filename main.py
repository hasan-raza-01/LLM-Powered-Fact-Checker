# FastAPI Backend for LLM-Powered Fact Checker
# Provides REST API endpoints for fact-checking

import sys
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.fact_checker import logging
from src.fact_checker.pipeline import DataIngestionPipeline, FactCheckingPipeline
from src.fact_checker.entity import FactCheckEntity


# Request/Response Models
class FactCheckRequest(BaseModel):
    """Request model for fact-checking endpoint"""
    claim: str


class FactCheckResponse(BaseModel):
    """Response model for fact-checking endpoint"""
    original_input: str
    claim: str
    verdict: str
    evidence: list[str]
    reasoning: str
    confidence_score: float | None = None


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    vectordb_status: str
    document_count: int


# Global state
app_state = {
    "fact_checker": None,
    "document_count": 0,
    "is_ready": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logging.info("=" * 60)
    logging.info("LLM-Powered Fact Checker - Starting Up")
    logging.info("=" * 60)
    
    try:
        # Step 1: Run Data Ingestion Pipeline
        logging.info("Step 1: Running Data Ingestion Pipeline...")
        ingestion_pipeline = DataIngestionPipeline()
        ingestion_result = ingestion_pipeline.run()
        app_state["document_count"] = ingestion_result.document_count
        logging.info(f"Data Ingestion complete: {ingestion_result.document_count} documents")
        
        # Step 2: Initialize Fact Checking Pipeline (lazy loading of models)
        logging.info("Step 2: Initializing Fact Checking Pipeline...")
        app_state["fact_checker"] = FactCheckingPipeline()
        
        # Step 3: Health Check - Test retrieval and LLM
        logging.info("Step 3: Running health check...")
        test_result = app_state["fact_checker"].run("Test query for health check")
        logging.info(f"Health check passed: verdict={test_result.verdict}")
        
        app_state["is_ready"] = True
        logging.info("=" * 60)
        logging.info("Application startup complete - Server is READY")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.exception(f"Startup failed: {e}")
        app_state["is_ready"] = False
    
    yield
    
    # Shutdown
    logging.info("Application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="LLM-Powered Fact Checker",
    description="A RAG-based fact-checking API that verifies claims against verified government data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns the status of the application and vector database.
    """
    return HealthResponse(
        status="ready" if app_state["is_ready"] else "not_ready",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        vectordb_status="connected" if app_state["is_ready"] else "disconnected",
        document_count=app_state["document_count"]
    )


@app.post("/check", response_model=FactCheckResponse, tags=["Fact Check"])
def check_fact(request: FactCheckRequest):
    """
    Fact-check a claim.
    
    Takes a claim as input, performs detection, extraction, retrieval,
    and verification to produce a verdict with evidence and reasoning.
    """
    if not app_state["is_ready"]:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Please wait for startup to complete."
        )
    
    if not request.claim or len(request.claim.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Claim cannot be empty"
        )
    
    try:
        logging.info(f"Received fact-check request: {request.claim[:100]}...")
        
        result: FactCheckEntity = app_state["fact_checker"].run(request.claim)
        
        return FactCheckResponse(
            original_input=result.original_input,
            claim=result.claim,
            verdict=result.verdict,
            evidence=result.evidence,
            reasoning=result.reasoning,
            confidence_score=result.confidence_score
        )
        
    except Exception as e:
        logging.exception(f"Error processing fact-check request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
