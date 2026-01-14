# Constants for LLM-Powered Fact Checker
# Each pipeline has its own constants class

from pydantic import BaseModel, Field
from pathlib import Path


class DataIngestionConstants(BaseModel):
    """Constants for Data Ingestion Pipeline"""
    
    # Paths
    ARTIFACTS_DIR: Path = Field(default=Path("artifacts"))
    CSV_FILE_PATH: Path = Field(default=Path("artifacts/verified_facts.csv"))
    CHROMA_DB_PATH: Path = Field(default=Path("artifacts/chroma_db"))
    
    # ChromaDB
    COLLECTION_NAME: str = Field(default="verified_facts")
    
    # Embedding Model (HuggingFace)
    EMBEDDING_MODEL_NAME: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    

class FactCheckingConstants(BaseModel):
    """Constants for Fact Checking Pipeline"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    
    # LLM Models (Ollama)
    EXTRACTION_MODEL: str = Field(default="gemma:7b")
    VERIFICATION_MODEL: str = Field(default="deepseek-r1:7b")
    
    # Claim Detection Model (HuggingFace)
    CLAIM_DETECTION_MODEL: str = Field(default="Nithiwat/bert-base_claimbuster")
    
    # Retrieval
    TOP_K_RESULTS: int = Field(default=3)
    
    # Prompts
    EXTRACTION_PROMPT: str = Field(default="""
You are a claim extraction assistant. Extract the main factual claims from the following text.
Return ONLY the key claims as a JSON array of strings.

Text: {input_text}

Output format: ["claim1", "claim2", ...]
""")
    
    VERIFICATION_PROMPT: str = Field(default="""
You are a fact-checking assistant. Compare the claim against the retrieved evidence and determine if the claim is True, False, or Unverifiable.

Claim: {claim}

Retrieved Evidence:
{evidence}

Analyze the claim against the evidence and respond in the following JSON format:
{{
    "verdict": "True" | "False" | "Unverifiable",
    "reasoning": "Your detailed explanation of why this verdict was chosen"
}}
""")


# Instantiate constants
DATA_INGESTION_CONSTANTS = DataIngestionConstants()
FACT_CHECKING_CONSTANTS = FactCheckingConstants()

__all__ = [
    "DataIngestionConstants",
    "FactCheckingConstants", 
    "DATA_INGESTION_CONSTANTS",
    "FACT_CHECKING_CONSTANTS"
]
