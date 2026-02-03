from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import logging
from .model_service import DNATransformerService
from .schemas import (
    DNASequenceRequest, BatchRequest, PredictionResponse, 
    BatchResponse, HealthResponse, ErrorResponse,
    AccuracyTestRequest, AccuracyTestResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🧬 DNA Species Classification API",
    description="AI-powered eDNA species identification using Tiny Transformer",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize model service on startup"""
    global model_service
    try:
        model_dir = os.getenv("MODEL_DIR", "models")
        model_service = DNATransformerService(model_dir)
        logger.info("🚀 DNA Transformer service started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint"""
    return {
        "message": "🧬 DNA Species Classification API",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs"
    }

# Add favicon route to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors"""
    return JSONResponse(status_code=204, content={})

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    
    health_info = model_service.health_check()
    
    return HealthResponse(
        status="healthy" if health_info["model_loaded"] else "unhealthy",
        **health_info
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sequence(request: DNASequenceRequest):
    """Classify a single DNA sequence"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    start_time = time.time()
    
    try:
        # Validate sequence
        if not request.sequence.strip():
            raise HTTPException(status_code=400, detail="Empty sequence provided")
        
        # Make prediction
        result = model_service.predict(request.sequence)
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            species=result["species"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_info=result["model_info"],
            sequence_id=request.sequence_id,
            processing_time=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest):
    """Classify multiple DNA sequences"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    start_time = time.time()
    
    try:
        # Extract sequences
        sequences = [seq.sequence for seq in request.sequences]
        sequence_ids = [seq.sequence_id for seq in request.sequences]
        
        # Make batch prediction
        results = model_service.batch_predict(sequences)
        
        # Process results
        predictions = []
        successful_count = 0
        species_counts = {}
        
        for i, result in enumerate(results):
            pred = PredictionResponse(
                species=result.get("species", "Unknown"),
                confidence=result.get("confidence", 0.0),
                probabilities=result.get("probabilities", {}),
                model_info=result.get("model_info", {}),
                sequence_id=sequence_ids[i] or f"seq_{i}",
                processing_time=0  # Individual timing not tracked in batch
            )
            predictions.append(pred)
            
            if result.get("species"):
                successful_count += 1
                species = result["species"]
                species_counts[species] = species_counts.get(species, 0) + 1
        
        total_time = time.time() - start_time
        
        return BatchResponse(
            predictions=predictions,
            summary={
                "total_sequences": len(sequences),
                "successful_predictions": successful_count,
                "failed_predictions": len(sequences) - successful_count,
                "unique_species_found": len(species_counts),
                "species_distribution": species_counts,
                "sequences_per_second": len(sequences) / total_time if total_time > 0 else 0
            },
            total_processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get model information"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    return model_service.health_check()

@app.post("/model/test-accuracy", response_model=AccuracyTestResponse, tags=["Model"])
async def test_model_accuracy(request: AccuracyTestRequest):
    """Test model accuracy with provided sequences and true labels"""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        result = model_service.evaluate_accuracy(
            test_sequences=request.sequences,
            true_labels=request.true_labels
        )
        return AccuracyTestResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Accuracy testing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during accuracy testing")

# FIXED: Exception handlers now return JSONResponse objects
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )
