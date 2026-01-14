# Fact Checking Component
# Performs claim detection, extraction, retrieval, and verification

import sys
import json
import re
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
import httpx
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from src.fact_checker import logging
from src.fact_checker.exception import CustomException
from src.fact_checker.entity import ClaimEntity, RetrievalEntity, FactCheckEntity
from src.fact_checker.config import FactCheckingConfigInstance
from src.fact_checker.constants import DATA_INGESTION_CONSTANTS


class FactCheckingComponents:
    """
    Component for fact-checking claims using RAG pipeline.
    
    Flow:
        1. Claim Detection (filter if input is claim-worthy)
        2. Claim Extraction (extract key claims using LLM)
        3. Retrieval (find similar facts from ChromaDB)
        4. Verification (compare claim with evidence using LLM)
    
    Usage:
        component = FactCheckingComponents()
        result = component.run("The claim to check...")
    """
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or FactCheckingConfigInstance
        self.claim_detector = None
        self.embedding_model = None
        self.chroma_collection = None
        
    def _load_claim_detector(self):
        """Load claim detection model from HuggingFace"""
        try:
            logging.info(f"Loading claim detection model: {self.config.claim_detection_model}")
            self.claim_detector = pipeline(
                "text-classification",
                model=self.config.claim_detection_model,
                truncation=True,
                max_length=512
            )
            logging.info("Claim detection model loaded successfully")
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def _load_embedding_model(self):
        """Load embedding model for retrieval"""
        try:
            model_name = DATA_INGESTION_CONSTANTS.EMBEDDING_MODEL_NAME
            logging.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logging.info("Embedding model loaded successfully")
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def _load_chromadb(self):
        """Connect to ChromaDB collection"""
        try:
            chroma_path = DATA_INGESTION_CONSTANTS.CHROMA_DB_PATH
            collection_name = DATA_INGESTION_CONSTANTS.COLLECTION_NAME
            
            logging.info(f"Connecting to ChromaDB at {chroma_path}")
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            self.chroma_collection = client.get_collection(name=collection_name)
            logging.info(f"Connected to collection '{collection_name}' with {self.chroma_collection.count()} documents")
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def _call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API for LLM inference"""
        try:
            url = f"{self.config.ollama_base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            logging.info(f"Calling Ollama model: {model}")
            
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def detect_claim(self, text: str) -> Tuple[bool, float]:
        """
        Detect if the input text contains a claim worth checking.
        
        Returns:
            Tuple[bool, float]: (is_claim_worthy, confidence_score)
        """
        try:
            if self.claim_detector is None:
                self._load_claim_detector()
            
            logging.info("Detecting claim worthiness...")
            result = self.claim_detector(text)[0]
            
            # ClaimBuster model returns labels like 'NFS' (non-factual), 'UFS' (unimportant factual), 'CFS' (check-worthy factual)
            label = result.get('label', '')
            score = result.get('score', 0.0)
            
            # CFS means check-worthy factual statement
            is_claim_worthy = 'CFS' in label.upper() or score > 0.5
            
            logging.info(f"Claim detection result: label={label}, score={score}, is_claim_worthy={is_claim_worthy}")
            return is_claim_worthy, score
            
        except Exception as e:
            logging.exception(e)
            # Default to True to allow processing even if detection fails
            return True, 0.5
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract key factual claims from input text using LLM.
        
        Returns:
            List[str]: List of extracted claims
        """
        try:
            logging.info("Extracting claims from input text...")
            
            prompt = self.config.extraction_prompt.format(input_text=text)
            response = self._call_ollama(self.config.extraction_model, prompt)
            
            # Parse JSON array from response
            # Try to find JSON array in the response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                claims = json.loads(match.group())
                logging.info(f"Extracted {len(claims)} claims")
                return claims
            else:
                # If no JSON found, use the original text as the claim
                logging.warning("Could not parse claims from LLM response, using original text")
                return [text]
                
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON from LLM response, using original text")
            return [text]
        except Exception as e:
            logging.exception(e)
            return [text]
    
    def retrieve_facts(self, claim: str) -> RetrievalEntity:
        """
        Retrieve relevant facts from ChromaDB.
        
        Returns:
            RetrievalEntity: Retrieved documents with scores
        """
        try:
            if self.embedding_model is None:
                self._load_embedding_model()
            if self.chroma_collection is None:
                self._load_chromadb()
            
            logging.info(f"Retrieving top-{self.config.top_k_results} facts for claim...")
            
            # Generate embedding for the claim
            query_embedding = self.embedding_model.encode([claim])[0]
            
            # Query ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.top_k_results,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = results.get('documents', [[]])[0]
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            # Convert distances to similarity scores (ChromaDB returns L2 distances)
            # Lower distance = higher similarity
            similarity_scores = [1 / (1 + d) for d in distances]
            sources = [m.get('source', 'Unknown') for m in metadatas]
            
            logging.info(f"Retrieved {len(documents)} relevant facts")
            
            return RetrievalEntity(
                query=claim,
                retrieved_documents=documents,
                similarity_scores=similarity_scores,
                sources=sources
            )
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)
    
    def verify_claim(self, claim: str, evidence: List[str]) -> Tuple[str, str]:
        """
        Verify claim against retrieved evidence using LLM.
        
        Returns:
            Tuple[str, str]: (verdict, reasoning)
        """
        try:
            logging.info("Verifying claim against evidence...")
            
            # Format evidence for prompt
            evidence_text = "\n".join([f"- {e}" for e in evidence])
            
            prompt = self.config.verification_prompt.format(
                claim=claim,
                evidence=evidence_text
            )
            
            response = self._call_ollama(self.config.verification_model, prompt)
            
            # Parse JSON response
            # Try to find JSON object in response
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    verdict = result.get('verdict', 'Unverifiable')
                    reasoning = result.get('reasoning', 'Unable to determine.')
                    logging.info(f"Verification result: {verdict}")
                    return verdict, reasoning
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract verdict from text
            response_lower = response.lower()
            if 'true' in response_lower and 'false' not in response_lower:
                return 'True', response
            elif 'false' in response_lower:
                return 'False', response
            else:
                return 'Unverifiable', response
                
        except Exception as e:
            logging.exception(e)
            return 'Unverifiable', f"Error during verification: {str(e)}"
    
    def run(self, input_text: str) -> FactCheckEntity:
        """
        Execute the complete fact-checking pipeline.
        
        Args:
            input_text: The claim or statement to fact-check
            
        Returns:
            FactCheckEntity: Complete fact-check result
        """
        try:
            logging.info("=" * 50)
            logging.info("Starting Fact Checking Pipeline")
            logging.info(f"Input: {input_text[:100]}...")
            logging.info("=" * 50)
            
            # Step 1: Detect if input is a claim
            is_claim_worthy, claim_score = self.detect_claim(input_text)
            
            if not is_claim_worthy:
                logging.info("Input is not claim-worthy, returning unverifiable")
                return FactCheckEntity(
                    original_input=input_text,
                    claim=input_text,
                    verdict="Unverifiable",
                    evidence=[],
                    reasoning="The input does not appear to contain a factual claim that can be verified.",
                    confidence_score=claim_score
                )
            
            # Step 2: Extract key claims
            claims = self.extract_claims(input_text)
            main_claim = claims[0] if claims else input_text
            
            # Step 3: Retrieve relevant facts
            retrieval_result = self.retrieve_facts(main_claim)
            
            # Step 4: Verify claim against evidence
            verdict, reasoning = self.verify_claim(
                main_claim, 
                retrieval_result.retrieved_documents
            )
            
            logging.info("Fact Checking Pipeline completed successfully")
            logging.info("=" * 50)
            
            return FactCheckEntity(
                original_input=input_text,
                claim=main_claim,
                verdict=verdict,
                evidence=retrieval_result.retrieved_documents,
                reasoning=reasoning,
                confidence_score=max(retrieval_result.similarity_scores) if retrieval_result.similarity_scores else 0.0
            )
            
        except Exception as e:
            logging.exception(e)
            raise CustomException(e, sys)


__all__ = ["FactCheckingComponents"]
