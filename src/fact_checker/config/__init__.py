# Config module - exports configuration classes

from .builder import (
    DataIngestionConfig,
    FactCheckingConfig,
    DataIngestionConfigInstance,
    FactCheckingConfigInstance
)

__all__ = [
    "DataIngestionConfig",
    "FactCheckingConfig",
    "DataIngestionConfigInstance",
    "FactCheckingConfigInstance"
]
