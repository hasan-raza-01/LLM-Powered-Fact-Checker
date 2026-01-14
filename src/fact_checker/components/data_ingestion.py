# Data Ingestion Component
# Loads verified facts from CSV, embeds them, and stores in ChromaDB

import csv
import sys
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.fact_checker import logging
from src.fact_checker.exception import CustomException
from src.fact_checker.entity import DataIngestionEntity
from src.fact_checker.config import DataIngestionConfigInstance


class DataIngestionComponents:
    """
    Component for ingesting verified facts into ChromaDB.
    
    Usage:
        component = DataIngestionComponents()
        result = component.run()
    """
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or DataIngestionConfigInstance
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
    def _load_csv_data(self) -> List[Dict]:
        """Load verified facts from CSV file"""
        try:
            logging.info(f"Loading data from {self.config.csv_file_path}")
            facts = []
            
            with open(self.config.csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    facts.append({
                        'id': row['id'],
                        'statement': row['statement'],
                        'source': row['source'],
                        'date': row['date'],
                        'category': row['category']
                    })
            
            logging.info(f"Loaded {len(facts)} facts from CSV")
            return facts
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model from HuggingFace"""
        try:
            logging.info(f"Loading embedding model: {self.config.embedding_model_name}")
            model = SentenceTransformer(self.config.embedding_model_name)
            logging.info("Embedding model loaded successfully")
            return model
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def _initialize_chromadb(self) -> chromadb.Collection:
        """Initialize ChromaDB and create/get collection"""
        try:
            logging.info(f"Initializing ChromaDB at {self.config.chroma_db_path}")
            
            # Ensure directory exists
            self.config.chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.config.chroma_db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "Verified facts from PIB for fact-checking"}
            )
            
            logging.info(f"ChromaDB collection '{self.config.collection_name}' ready")
            return collection
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def _embed_and_store(self, facts: List[Dict]) -> int:
        """Embed facts and store in ChromaDB"""
        try:
            logging.info("Embedding and storing facts in ChromaDB")
            
            # Check if collection already has data
            existing_count = self.collection.count()
            if existing_count > 0:
                logging.info(f"Collection already has {existing_count} documents. Skipping ingestion.")
                return existing_count
            
            # Prepare data for embedding
            documents = [fact['statement'] for fact in facts]
            ids = [f"fact_{fact['id']}" for fact in facts]
            metadatas = [
                {
                    'source': fact['source'],
                    'date': fact['date'],
                    'category': fact['category']
                }
                for fact in facts
            ]
            
            # Generate embeddings
            logging.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )
            
            logging.info(f"Successfully stored {len(documents)} facts in ChromaDB")
            return len(documents)
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def run(self) -> DataIngestionEntity:
        """
        Execute the data ingestion pipeline.
        
        Returns:
            DataIngestionEntity: Result of the ingestion process
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Data Ingestion Pipeline")
            logging.info("=" * 50)
            
            # Step 1: Load CSV data
            facts = self._load_csv_data()
            
            # Step 2: Load embedding model
            self.embedding_model = self._load_embedding_model()
            
            # Step 3: Initialize ChromaDB
            self.collection = self._initialize_chromadb()
            
            # Step 4: Embed and store
            doc_count = self._embed_and_store(facts)
            
            logging.info("Data Ingestion Pipeline completed successfully")
            logging.info("=" * 50)
            
            return DataIngestionEntity(
                chroma_db_path=self.config.chroma_db_path,
                collection_name=self.config.collection_name,
                document_count=doc_count,
                embedding_model=self.config.embedding_model_name,
                status="success"
            )
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


__all__ = ["DataIngestionComponents"]
