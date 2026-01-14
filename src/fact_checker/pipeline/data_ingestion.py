# Data Ingestion Pipeline
# Orchestrates the data ingestion process

from src.fact_checker import logging
from src.fact_checker.components import DataIngestionComponents
from src.fact_checker.config import DataIngestionConfigInstance
from src.fact_checker.entity import DataIngestionEntity


class DataIngestionPipeline:
    """
    Pipeline for Data Ingestion process.
    
    Loads verified facts from CSV, embeds them using Qwen model,
    and stores in ChromaDB for later retrieval.
    
    Usage:
        pipeline = DataIngestionPipeline()
        result = pipeline.run()
    """
    
    def __init__(self):
        self.config = DataIngestionConfigInstance
        
    def run(self) -> DataIngestionEntity:
        """
        Execute the data ingestion pipeline.
        
        Returns:
            DataIngestionEntity: Result containing ChromaDB path and document count
        """
        logging.info("Initializing Data Ingestion Pipeline")
        
        component = DataIngestionComponents(self.config)
        result = component.run()
        
        logging.info(f"Data Ingestion Pipeline completed: {result.document_count} documents ingested")
        
        return result


__all__ = ["DataIngestionPipeline"]
