from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import requests
import faiss
import json
import yaml
import numpy as np
import google.generativeai as genai
import os
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG_PATH = "/app/config.yaml"
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded.")
except Exception as e:
    logging.error(f"Error loading configuration from {CONFIG_PATH}: {e}")
    config = {} # Use defaults or fail gracefully

FAISS_INDEX_PATH = config.get('faiss_index_path', '/app/data/faiss_index/code_index.faiss')
FAISS_METADATA_PATH = config.get('faiss_metadata_path', '/app/data/faiss_index/code_metadata.json')
REPO_CLONE_DIR = config.get('repo_clone_dir', '/app/data/cloned_repos') # Needed to read context file content
TOP_K = config.get('top_k_similar', 3)
MAX_CONTEXT_LENGTH = config.get('max_context_length', 3500)
LLM_MODEL_NAME = config.get('llm_model_name', 'gemini-pro')
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://embedding-server:8000/embed") # Get from env or default

# --- Gemini API Key ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set.")
    # Depending on policy, you might exit or run with LLM disabled
    # raise RuntimeError("GEMINI_API_KEY not set.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Gemini API configured.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        # Handle appropriately

app = FastAPI()

# --- Globals (Load index and metadata on startup) ---
faiss_index = None
metadata = []

@app.on_event("startup")
async def startup_event():
    global faiss_index, metadata
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            logging.error(f"Faiss index file not found at {FAISS_INDEX_PATH}. Run indexing first.")
            # Optional: prevent startup if index missing
            # raise RuntimeError("Faiss index not found.")
            return # Allow startup but endpoints will fail

        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        logging.info(f"Faiss index loaded with {faiss_index.ntotal} vectors.")

        if not os.path.exists(FAISS_METADATA_PATH):
             logging.error(f"Metadata file not found at {FAISS_METADATA_PATH}.")
             # Optional: prevent startup if metadata missing
             # raise RuntimeError("Metadata file not found.")
             return

        with open(FAISS_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Metadata loaded for {len(metadata)} files.")

        if faiss_index is not None and len(metadata) != faiss_index.ntotal:
            logging.warning("Mismatch between Faiss index size and metadata count!")

    except Exception as e:
        logging.error(f"Failed to load Faiss index or metadata during startup: {e}")
        # Optional: prevent startup on error
        # raise RuntimeError(f"Startup failed: {e}")

# --- Request/Response Models ---
class CheckRequest(BaseModel):
    code: str

class CheckResponse(BaseModel):
    is_plagiarized: bool
    references: Optional[List[str]] = None
    error: Optional[str] = None
    # Optional: Add similarity scores if needed for debugging/evaluation
    # similarity_scores: Optional[List[float]] = None

# --- Helper Functions ---
def get_embedding_from_server(code_text: str) -> Optional[np.ndarray]:
    try:
        response = requests.post(EMBEDDING_SERVER_URL, json={"code": code_text}, timeout=10) # Added timeout
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
        embedding_list = response.json().get("embedding")
        if embedding_list:
            return np.array(embedding_list).astype('float32').reshape(1, -1) # Reshape for Faiss
        else:
            logging.error("Embedding not found in response from embedding server.")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling embedding server at {EMBEDDING_SERVER_URL}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error getting embedding: {e}")
        return None

def search_similar_code(query_embedding: np.ndarray) -> tuple[list[float], list[int]]:
    if faiss_index is None or query_embedding is None:
        logging.error("Faiss index or query embedding is not available for search.")
        return [], []
    try:
        # Ensure embedding is float32 and 2D
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype('float32')
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        # Search the index
        k = min(TOP_K, faiss_index.ntotal) # Adjust k if index is smaller than TOP_K
        distances, indices = faiss_index.search(query_embedding, k)
        return distances[0].tolist(), indices[0].tolist() # Return distances and indices for the first query
    except Exception as e:
        logging.error(f"Error during Faiss search: {e}")
        return [], []

def truncate_text(text: str, max_len: int) -> str:
    """Naive truncation"""
    if len(text) > max_len:
        return text[:max_len // 2] + "\n...\n" + text[-max_len // 2:]
    return text

def call_llm_for_plagiarism(user_code: str, similar_files_content: List[dict]) -> tuple[bool, List[str]]:
    if not GEMINI_API_KEY or not similar_files_content:
        logging.warning("LLM check skipped: API key missing or no similar files found.")
        return False, [] # Default to not plagiarized if LLM can't be called or no context

    prompt_parts = [
        "You are a code plagiarism detection assistant.",
        "Analyze if the following 'User Code Snippet' is plagiarized from any of the 'Reference Code Files' provided below.",
        "Respond with ONLY 'yes' or 'no'.",
        "If your answer is 'yes', ALSO provide the file path(s) from the references that the user code is plagiarized from, listing each relevant file path on a new line after 'References:'.",
        "Focus only on substantial similarity, ignoring minor differences like variable names or comments if the core logic/structure is the same.",
        "\n---\nUser Code Snippet:\n---\n",
        truncate_text(user_code, MAX_CONTEXT_LENGTH // (len(similar_files_content) + 1) if similar_files_content else MAX_CONTEXT_LENGTH), # Allocate budget
        "\n---\nReference Code Files:\n---"
    ]

    total_ref_length = 0
    for file_info in similar_files_content:
        # Adjust truncation budget based on number of files and user code length
        allocated_length = (MAX_CONTEXT_LENGTH - len(prompt_parts[5])) // max(1, len(similar_files_content))
        truncated_content = truncate_text(file_info['content'], allocated_length)
        prompt_parts.append(f"\nFile Path: {file_info['path']}\n```\n{truncated_content}\n```\n")
        total_ref_length += len(truncated_content) # Track length for logging/debugging

    final_prompt = "\n".join(prompt_parts)
    # logging.info(f"LLM Prompt (approx length {len(final_prompt)}):\n{final_prompt[:500]}...") # Log truncated prompt

    try:
        model = genai.GenerativeModel(LLM_MODEL_NAME)
        response = model.generate_content(final_prompt)

        # --- Robust Parsing of LLM Response ---
        response_text = response.text.strip().lower()
        logging.info(f"LLM Raw Response: '{response_text[:200]}...'")

        is_plagiarized = response_text.startswith("yes")
        references = []

        if is_plagiarized:
            # Look for references after "yes" or "references:" marker
            ref_section_marker = "references:"
            ref_start_index = response_text.find(ref_section_marker)
            if ref_start_index != -1:
                ref_text = response_text[ref_start_index + len(ref_section_marker):].strip()
                # Split by newline and filter potential empty strings
                references = [ref.strip() for ref in ref_text.split('\n') if ref.strip()]
            else:
                # Fallback: try to extract paths directly if "references:" is missing but starts with "yes"
                lines_after_yes = response_text.split('\n')[1:] # Skip the "yes" line
                references = [line.strip() for line in lines_after_yes if line.strip() and '/' in line] # Simple path heuristic

            # Validate extracted references against provided context paths (optional but good)
            valid_references = [ref for ref in references if any(ref == ctx['path'] for ctx in similar_files_content)]
            if not valid_references and references: # If LLM listed paths, but none match context
                 logging.warning(f"LLM listed references not found in context: {references}")
                 # Decide policy: keep LLM refs, or clear them? Clearing might be safer.
                 # references = []
            elif valid_references:
                references = valid_references # Use only validated refs

        logging.info(f"LLM Parsed - Plagiarized: {is_plagiarized}, References: {references}")
        return is_plagiarized, references

    except Exception as e:
        logging.error(f"Error calling LLM or parsing response: {e}")
        # Fallback on error: assume not plagiarized
        return False, []

# --- API Endpoint ---
@app.post("/check_plagiarism", response_model=CheckResponse)
async def check_plagiarism(request: CheckRequest):
    if faiss_index is None or not metadata:
         raise HTTPException(status_code=503, detail="Index or metadata not loaded. Service unavailable.")
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=400, detail="Code snippet cannot be empty")

    logging.info("Received plagiarism check request.")

    # 1. Get embedding for the user's code
    query_embedding = get_embedding_from_server(request.code)
    if query_embedding is None:
        return CheckResponse(is_plagiarized=False, error="Failed to get embedding for the provided code.")

    # 2. Search Faiss index
    distances, indices = search_similar_code(query_embedding)
    if not indices:
        # If no similar files found (e.g., index empty or search error)
         logging.info("No similar documents found in Faiss index.")
         # Call LLM without context? Or return directly? Let's return directly for MVP.
         # If calling LLM w/o context: is_plagiarized, _ = call_llm_for_plagiarism(request.code, [])
         return CheckResponse(is_plagiarized=False, references=[]) # Assume not plagiarized if no matches

    logging.info(f"Found top {len(indices)} similar files with indices: {indices} and distances: {distances}")

    # 3. Retrieve content of similar files
    similar_files_content = []
    retrieved_paths = []
    for idx in indices:
        if 0 <= idx < len(metadata):
            file_info = metadata[idx]
            file_path = file_info.get("file_path")
            if file_path:
                full_path = os.path.join(REPO_CLONE_DIR, file_path)
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    similar_files_content.append({"path": file_path, "content": content})
                    retrieved_paths.append(file_path)
                except FileNotFoundError:
                    logging.warning(f"Metadata points to file not found: {full_path}")
                except Exception as e:
                    logging.error(f"Error reading file {full_path}: {e}")
        else:
             logging.warning(f"Invalid index {idx} returned from Faiss search.")

    if not similar_files_content:
        logging.warning("Could not retrieve content for any similar files found.")
        # Decide how to handle - maybe still call LLM? For now, return non-plagiarized.
        return CheckResponse(is_plagiarized=False, references=[])

    # 4. Call LLM with user code and context
    is_plagiarized, references = call_llm_for_plagiarism(request.code, similar_files_content)

    # Ensure references are only returned if plagiarized is true
    final_references = references if is_plagiarized else []

    # Optional: filter references to only include paths initially retrieved
    # final_references = [ref for ref in final_references if ref in retrieved_paths]

    logging.info(f"Final result - Plagiarized: {is_plagiarized}, References: {final_references}")
    return CheckResponse(is_plagiarized=is_plagiarized, references=final_references)


@app.get("/health")
async def health_check():
    # Basic health check
    index_loaded = faiss_index is not None
    metadata_loaded = bool(metadata)
    # Check embedding server health (optional but good)
    emb_server_status = "unknown"
    try:
        response = requests.get(f"{EMBEDDING_SERVER_URL.replace('/embed','/health')}", timeout=2)
        if response.status_code == 200:
            emb_server_status = response.json().get("status", "error")
        else:
            emb_server_status = f"error_{response.status_code}"
    except Exception:
        emb_server_status = "unreachable"

    return {
        "status": "ok" if index_loaded and metadata_loaded else "degraded",
        "index_loaded": index_loaded,
        "index_vector_count": faiss_index.ntotal if index_loaded else 0,
        "metadata_loaded": metadata_loaded,
        "metadata_file_count": len(metadata) if metadata_loaded else 0,
        "embedding_server_status": emb_server_status,
        "gemini_configured": GEMINI_API_KEY is not None
    }

# For local testing without uvicorn command line
# if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8001)