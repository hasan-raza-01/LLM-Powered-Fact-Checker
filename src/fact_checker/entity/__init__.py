# Entity classes - Return types for each pipeline
# These are Pydantic models that define the output structure of components

from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path


class DataIngestionEntity(BaseModel):
    """Return type for Data Ingestion Component"""
    
    chroma_db_path: Path
    collection_name: str
    document_count: int
    embedding_model: str
    status: str = Field(default="success")


class ClaimEntity(BaseModel):
    """Entity for extracted claims"""
    
    original_text: str
    extracted_claims: List[str]
    is_claim_worthy: bool
    claim_score: float = Field(default=0.0)


class RetrievalEntity(BaseModel):
    """Entity for retrieved facts from vector database"""
    
    query: str
    retrieved_documents: List[str]
    similarity_scores: List[float]
    sources: List[str]


class FactCheckEntity(BaseModel):
    """Return type for Fact Checking Pipeline - Final Output"""
    
    original_input: str
    claim: str
    verdict: str  # "True", "False", or "Unverifiable"
    evidence: List[str]
    reasoning: str
    confidence_score: Optional[float] = None


__all__ = [
    "DataIngestionEntity",
    "ClaimEntity", 
    "RetrievalEntity",
    "FactCheckEntity"
]
