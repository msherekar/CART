from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging
import os
import torch
from pathlib import Path

# Get project root using os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import AutoTokenizer, AutoModelForMaskedLM
from CART.src.utils.sequence_utils import validate_fasta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CART Cytotoxicity Prediction API",
    description="API for predicting cytotoxicity scores of protein(CAR) sequences using high and low models",
    version="1.0.0"
)

# Load models at startup
high_model = None
low_model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    global high_model, low_model, tokenizer
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
        
        # Load high model
        high_model_path = os.path.join(project_root, "models", "high_best.pth")
        if not os.path.exists(high_model_path):
            raise FileNotFoundError(f"High model not found at {high_model_path}")
        high_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        high_model.load_state_dict(torch.load(high_model_path, map_location=device))
        high_model.to(device)
        high_model.eval()
        logger.info("High model loaded successfully")
        
        # Load low model
        low_model_path = os.path.join(project_root, "models", "low_best.pth")
        if not os.path.exists(low_model_path):
            raise FileNotFoundError(f"Low model not found at {low_model_path}")
        low_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        low_model.load_state_dict(torch.load(low_model_path, map_location=device))
        low_model.to(device)
        low_model.eval()
        logger.info("Low model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

class SequenceRequest(BaseModel):
    sequence: str
    description: str = ""

class PredictionResponse(BaseModel):
    sequence: str
    description: str
    high_model_score: float
    low_model_score: float
    average_score: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_sequence(request: SequenceRequest):
    """
    Predict cytotoxicity scores for a given protein sequence using both high and low models.
    
    Args:
        request: SequenceRequest containing the protein sequence and optional description
        
    Returns:
        PredictionResponse with scores from both models and their average
    """
    try:
        # Validate FASTA format
        if not validate_fasta(request.sequence):
            raise HTTPException(status_code=400, detail="Invalid FASTA format")
        
        # Extract sequence from FASTA format
        sequence = request.sequence.split('\n', 1)[1].replace('\n', '')
        
        # Tokenize sequence
        inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions from both models
        with torch.no_grad():
            high_outputs = high_model(**inputs)
            low_outputs = low_model(**inputs)
            high_score = high_outputs.logits.mean().item()
            low_score = low_outputs.logits.mean().item()
        
        # Calculate average score
        avg_score = (high_score + low_score) / 2
        
        return PredictionResponse(
            sequence=request.sequence,
            description=request.description,
            high_model_score=high_score,
            low_model_score=low_score,
            average_score=avg_score
        )
        
    except Exception as e:
        logger.error(f"Error processing sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and models are loaded.
    """
    if high_model is None or low_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True} 