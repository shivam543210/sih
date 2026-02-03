from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DNASequenceRequest(BaseModel):
    sequence: str = Field(..., min_length=10, description="DNA sequence (A, T, G, C)")
    sequence_id: Optional[str] = Field(None, description="Optional sequence identifier")

class BatchRequest(BaseModel):
    sequences: List[DNASequenceRequest] = Field(..., max_items=100, description="List of DNA sequences")

class PredictionResponse(BaseModel):
    species: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_info: Dict
    sequence_id: Optional[str] = None
    processing_time: Optional[float] = None

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict
    total_processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    supported_species: List[str]
    k_mer_size: int
    max_sequence_length: int

class ErrorResponse(BaseModel):
    error: str
    detail: str

class AccuracyTestRequest(BaseModel):
    sequences: List[str] = Field(..., description="List of DNA sequences for testing")
    true_labels: List[str] = Field(..., description="List of true species labels")

class AccuracyTestResponse(BaseModel):
    accuracy: float = Field(..., ge=0.0, le=1.0)
    total_sequences: int
    correct_predictions: int
    predictions: List[Dict]
    evaluation_summary: Dict
