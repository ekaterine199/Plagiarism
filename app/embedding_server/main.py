from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG_PATH = "/app/config.yaml"
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded.")
except Exception as e:
    logging.error(f"Error loading configuration from {CONFIG_PATH}: {e}")
    # Fallback or default model name if config fails
    config = {'embedding_model': 'microsoft/codebert-base'}


EMBEDDING_MODEL = config.get('embedding_model', 'microsoft/codebert-base')

app = FastAPI()

# --- Globals (Load model on startup) ---
tokenizer = None
model = None
device = None

@app.on_event("startup")
async def startup_event():
    global tokenizer, model, device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL, from_tf=True).to(device)
        model.eval() # Set model to evaluation mode
        logging.info(f"Embedding model '{EMBEDDING_MODEL}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model during startup: {e}")
        # You might want to prevent the app from starting fully if the model fails
        raise RuntimeError(f"Could not load embedding model: {e}")

# --- Request Model ---
class EmbedRequest(BaseModel):
    code: str

# --- Helper Function ---
def get_embedding(code_text):
    if not model or not tokenizer:
         raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Using mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding.tolist() # Return as list for JSON serialization
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

# --- API Endpoint ---
@app.post("/embed")
async def embed_code(request: EmbedRequest):
    if not request.code or not request.code.strip():
         raise HTTPException(status_code=400, detail="Code snippet cannot be empty")
    logging.info(f"Received embedding request for code snippet (length: {len(request.code)}).")
    embedding = get_embedding(request.code)
    logging.info("Embedding generated successfully.")
    return {"embedding": embedding}

@app.get("/health")
async def health_check():
    # Basic health check
    return {"status": "ok", "model_loaded": model is not None}

# For local testing without uvicorn command line
# if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)