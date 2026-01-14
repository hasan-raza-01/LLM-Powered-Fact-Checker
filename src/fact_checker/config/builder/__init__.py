# Config Builder - Creates configuration objects from constants and config.yaml

from ...constants import DATA_INGESTION_CONSTANTS, FACT_CHECKING_CONSTANTS
from ...entity import DataIngestionEntity, FactCheckEntity
from pathlib import Path


class DataIngestionConfig:
    """Configuration for Data Ingestion Pipeline"""
    
    def __init__(self):
        self.constants = DATA_INGESTION_CONSTANTS
        
    @property
    def artifacts_dir(self) -> Path:
        return self.constants.ARTIFACTS_DIR
    
    @property
    def csv_file_path(self) -> Path:
        return self.constants.CSV_FILE_PATH
    
    @property
    def chroma_db_path(self) -> Path:
        return self.constants.CHROMA_DB_PATH
    
    @property
    def collection_name(self) -> str:
        return self.constants.COLLECTION_NAME
    
    @property
    def embedding_model_name(self) -> str:
        return self.constants.EMBEDDING_MODEL_NAME


class FactCheckingConfig:
    """Configuration for Fact Checking Pipeline"""
    
    def __init__(self):
        self.constants = FACT_CHECKING_CONSTANTS
        
    @property
    def ollama_base_url(self) -> str:
        return self.constants.OLLAMA_BASE_URL
    
    @property
    def extraction_model(self) -> str:
        return self.constants.EXTRACTION_MODEL
    
    @property
    def verification_model(self) -> str:
        return self.constants.VERIFICATION_MODEL
    
    @property
    def claim_detection_model(self) -> str:
        return self.constants.CLAIM_DETECTION_MODEL
    
    @property
    def top_k_results(self) -> int:
        return self.constants.TOP_K_RESULTS
    
    @property
    def extraction_prompt(self) -> str:
        return self.constants.EXTRACTION_PROMPT
    
    @property
    def verification_prompt(self) -> str:
        return self.constants.VERIFICATION_PROMPT


# Create config instances
DataIngestionConfigInstance = DataIngestionConfig()
FactCheckingConfigInstance = FactCheckingConfig()

__all__ = [
    "DataIngestionConfig",
    "FactCheckingConfig",
    "DataIngestionConfigInstance",
    "FactCheckingConfigInstance"
]
