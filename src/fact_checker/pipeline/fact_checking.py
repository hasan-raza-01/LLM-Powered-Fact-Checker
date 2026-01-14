# Fact Checking Pipeline  
# Orchestrates the complete fact-checking process

from src.fact_checker import logging
from src.fact_checker.components import FactCheckingComponents
from src.fact_checker.config import FactCheckingConfigInstance
from src.fact_checker.entity import FactCheckEntity


class FactCheckingPipeline:
    """
    Pipeline for Fact Checking process.
    
    Takes input claim, performs detection, extraction, retrieval,
    and verification to produce a verdict with evidence.
    
    Usage:
        pipeline = FactCheckingPipeline()
        result = pipeline.run("The claim to check...")
    """
    
    def __init__(self):
        self.config = FactCheckingConfigInstance
        self._component = None
        
    @property
    def component(self) -> FactCheckingComponents:
        """Lazy initialization of component for model loading"""
        if self._component is None:
            self._component = FactCheckingComponents(self.config)
        return self._component
        
    def run(self, claim: str) -> FactCheckEntity:
        """
        Execute the fact-checking pipeline.
        
        Args:
            claim: The statement/claim to fact-check
            
        Returns:
            FactCheckEntity: Result containing verdict, evidence, and reasoning
        """
        logging.info("Initializing Fact Checking Pipeline")
        logging.info(f"Claim: {claim[:100]}...")
        
        result = self.component.run(claim)
        
        logging.info(f"Fact Checking Pipeline completed: verdict={result.verdict}")
        
        return result


__all__ = ["FactCheckingPipeline"]
