import requests
import pandas as pd
import json
import yaml
import os
import time
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import faiss # For RAG-only local search
import google.generativeai as genai # For LLM-only

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
CONFIG_PATH = "/app/config.yaml"
TEST_DATA_PATH = "/app/test_data.json" # Path inside the container
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded.")
except Exception as e:
    logging.error(f"Error loading configuration from {CONFIG_PATH}: {e}")
    config = {}

CHECKER_API_URL = os.getenv("CHECKER_API_URL", "http://plagiarism-checker:8001/check_plagiarism")
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", "http://embedding-server:8000/embed") # Needed for RAG-only if local embedding
EVALUATION_OUTPUT_CSV = config.get('evaluation_output_csv', '/app/data/evaluation_results.csv')
RAG_ONLY_THRESHOLD = config.get('evaluation_rag_threshold', 0.95)
FAISS_INDEX_PATH = config.get('faiss_index_path', '/app/data/faiss_index/code_index.faiss')
FAISS_METADATA_PATH = config.get('faiss_metadata_path', '/app/data/faiss_index/code_metadata.json')
TOP_K = config.get('top_k_similar', 3)
LLM_MODEL_NAME = config.get('llm_model_name', 'gemini-pro')

# --- Gemini API Key (for LLM-Only) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not set. LLM-Only approach will be skipped.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Gemini API configured for LLM-Only evaluation.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        GEMINI_API_KEY = None # Disable LLM-Only

# --- Load Faiss Index and Metadata (for RAG-Only) ---
# Option 1: Load locally (requires faiss, model deps in this container)
# Option 2: Call embedding server and plagiarism checker's search logic (more complex)
# Choosing Option 1 for simplicity here, assuming index/metadata available via volume
faiss_index = None
metadata = []
embeddings_np = None # Store original embeddings if needed for cosine sim
try:
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        # If using IndexFlatIP, no need to reload embeddings.
        # If using IndexFlatL2, we need original vectors for cosine similarity
        # This requires either storing them separately or reconstructing from the index if possible (not directly with IndexFlatL2)
        # Simplification: Assume we can get embeddings from the embedding server
        logging.info(f"Faiss index loaded for RAG-Only mode with {faiss_index.ntotal} vectors.")
    else:
        logging.warning(f"Faiss index not found at {FAISS_INDEX_PATH}. RAG-Only mode might fail.")

    if os.path.exists(FAISS_METADATA_PATH):
        with open(FAISS_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Metadata loaded for {len(metadata)} files.")
    else:
        logging.warning(f"Metadata file not found at {FAISS_METADATA_PATH}. RAG-Only mode might fail.")

except Exception as e:
    logging.error(f"Error loading Faiss/metadata for RAG-Only mode: {e}")
    faiss_index = None # Disable RAG-Only if loading fails

# --- Load Test Data ---
try:
    with open(TEST_DATA_PATH, 'r') as f:
        test_data = json.load(f)
    logging.info(f"Loaded {len(test_data)} examples from {TEST_DATA_PATH}")
except FileNotFoundError:
    logging.error(f"Test data file not found at {TEST_DATA_PATH}")
    exit(1)
except json.JSONDecodeError:
    logging.error(f"Error decoding JSON from {TEST_DATA_PATH}")
    exit(1)

# --- Evaluation Functions ---

def evaluate_full_system(snippet):
    """Calls the main plagiarism checking API."""
    try:
        response = requests.post(CHECKER_API_URL, json={"code": snippet}, timeout=30) # Longer timeout for full check
        response.raise_for_status()
        data = response.json()
        return data.get("is_plagiarized", False), data.get("references", [])
    except requests.exceptions.Timeout:
        logging.error(f"Timeout calling Full System API for snippet id")
        return False, [] # Default on timeout
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Full System API: {e}")
        return False, [] # Default on error
    except Exception as e:
        logging.error(f"Unexpected error in evaluate_full_system: {e}")
        return False, []

def evaluate_rag_only(snippet):
    """Uses local Faiss search + cosine similarity threshold."""
    if faiss_index is None or not metadata:
         logging.warning("Skipping RAG-Only: Index or metadata not loaded.")
         return False, []

    # 1. Get embedding (call server or load model locally - calling server here)
    try:
        emb_response = requests.post(EMBEDDING_SERVER_URL, json={"code": snippet}, timeout=10)
        emb_response.raise_for_status()
        embedding_list = emb_response.json().get("embedding")
        if not embedding_list:
            logging.error("RAG-Only: Failed to get embedding.")
            return False, []
        query_embedding = np.array(embedding_list).astype('float32').reshape(1, -1)
    except Exception as e:
        logging.error(f"RAG-Only: Error getting embedding: {e}")
        return False, []

    # 2. Search Faiss
    try:
        k = min(TOP_K, faiss_index.ntotal)
        distances, indices = faiss_index.search(query_embedding, k)

        if indices[0][0] == -1: # Check if any results were found
             logging.info("RAG-Only: No similar docs found in Faiss.")
             return False, []

        # 3. Calculate Cosine Similarity (Requires original vectors if using L2 index)
        # We'll use the L2 distance as a proxy or re-implement cosine here.
        # For simplicity, let's just check if the *closest* match distance is very small (proxy for high L2 similarity)
        # OR - calculate actual cosine similarity if we have the vectors
        # Let's try calculating cosine similarity using the embedding server for reference vectors (less efficient but feasible for eval)

        top_indices = indices[0]
        top_distances = distances[0]
        references = []
        max_similarity = -1.0

        # Fetch embeddings for top k results to calculate cosine similarity
        top_k_embeddings = []
        valid_indices = []
        original_paths = []

        # We need the original indexed files to get their embeddings again for cosine sim.
        # This is inefficient. A better RAG-only might use IndexFlatIP directly.
        # Let's stick to a simpler L2 distance proxy for this MVP eval:
        # If L2 distance is small, similarity is high. Need to normalize?
        # Faiss L2 distance is squared Euclidean. Lower is better.
        # Let's define a threshold based on observed distances for near-duplicates.
        # This threshold is highly empirical.
        # A more robust way: Use `IndexFlatIP` and normalize vectors during indexing.
        # Then the dot product *is* cosine similarity.

        # Sticking with L2 proxy: Define an empirical L2 distance threshold.
        # It's hard to set a universal L2 threshold corresponding to 0.95 cosine sim.
        # Let's *assume* IndexFlatIP was used for simplicity of evaluation code.
        # If using IndexFlatL2, this part needs adjustment/re-indexing.
        # ASSUMING IndexFlatIP: distances are dot products (higher is better)
        # We need to find if max_similarity > RAG_ONLY_THRESHOLD

        # Reverting to L2 distance check as IndexFlatL2 is simpler to implement initially
        min_distance = top_distances[0] if len(top_distances) > 0 else float('inf')
        # We need a threshold for L2 distance. Let's guess a small value.
        # Lower distance means more similar.
        # This is NOT cosine similarity and less reliable.
        l2_threshold = 0.5 # VERY EMPIRICAL - ADJUST BASED ON OBSERVATION
        is_plagiarized = min_distance < l2_threshold

        if is_plagiarized:
            # Get the path of the most similar item
             closest_idx = top_indices[0]
             if 0 <= closest_idx < len(metadata):
                 references = [metadata[closest_idx].get("file_path", "unknown_path")]


        # # Cosine Similarity Calculation (if embeddings available) - More complex
        # for i, idx in enumerate(top_indices):
        #      if idx != -1 and 0 <= idx < len(metadata):
        #          # Need the embedding vector for idx - how to get it?
        #          # Option A: Reconstruct (not easy for IndexFlatL2)
        #          # Option B: Store embeddings separately (memory)
        #          # Option C: Re-embed the original file (slow)
        #          # Let's skip accurate cosine sim for L2 index in this MVP evaluation
        #          pass # Placeholder

        # Using L2 distance proxy result
        logging.info(f"RAG-Only: Min L2 distance {min_distance:.4f}. Plagiarized: {is_plagiarized}")
        return is_plagiarized, references

    except Exception as e:
        logging.error(f"RAG-Only: Error during search or similarity calc: {e}")
        return False, []

def evaluate_llm_only(snippet):
    """Calls the LLM directly with a simple prompt."""
    if not GEMINI_API_KEY:
        return False, [] # Skip if key missing

    prompt = (
        "Is the following code snippet plagiarized? "
        "Focus on whether it seems copied from common sources or is standard boilerplate. "
        "Answer ONLY 'yes' or 'no'."
        "\n\nCode Snippet:\n```\n"
        f"{snippet}"
        "\n```"
    )

    try:
        model = genai.GenerativeModel(LLM_MODEL_NAME)
        response = model.generate_content(prompt)
        response_text = response.text.strip().lower()
        logging.info(f"LLM-Only Raw Response: '{response_text}'")
        is_plagiarized = response_text.startswith("yes")
        return is_plagiarized, [] # LLM-only has no references in this setup
    except Exception as e:
        logging.error(f"Error calling LLM-Only: {e}")
        return False, []


# --- Main Evaluation Loop ---
results = []
for i, example in enumerate(test_data):
    logging.info(f"--- Processing Example {i+1}/{len(test_data)} (ID: {example.get('id', 'N/A')}) ---")
    snippet = example["snippet"]
    ground_truth = example["is_plagiarized_ground_truth"]
    # expected_refs = example.get("expected_references", []) # Not used for metrics yet

    # Run Full System
    start_time = time.time()
    fs_plagiarized, fs_refs = evaluate_full_system(snippet)
    fs_time = time.time() - start_time
    logging.info(f"Full System: Plagiarized={fs_plagiarized}, Time={fs_time:.2f}s")

    # Run RAG Only
    start_time = time.time()
    rag_plagiarized, rag_refs = evaluate_rag_only(snippet)
    rag_time = time.time() - start_time
    logging.info(f"RAG Only: Plagiarized={rag_plagiarized}, Time={rag_time:.2f}s")


    # Run LLM Only
    start_time = time.time()
    llm_plagiarized, llm_refs = evaluate_llm_only(snippet)
    llm_time = time.time() - start_time
    logging.info(f"LLM Only: Plagiarized={llm_plagiarized}, Time={llm_time:.2f}s")


    results.append({
        "id": example.get('id', i),
        "ground_truth": ground_truth,
        "full_system_pred": fs_plagiarized,
        "full_system_time": fs_time,
        "rag_only_pred": rag_plagiarized,
        "rag_only_time": rag_time,
        "llm_only_pred": llm_plagiarized,
        "llm_only_time": llm_time,
    })
    time.sleep(1) # Avoid hitting rate limits, adjust as needed

# --- Calculate Metrics ---
results_df = pd.DataFrame(results)
y_true = results_df["ground_truth"]

metrics = {}
for approach in ["full_system", "rag_only", "llm_only"]:
    y_pred = results_df[f"{approach}_pred"]
    accuracy = accuracy_score(y_true, y_pred)
    # Use zero_division=0 to handle cases with no predicted positives
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    metrics[approach] = {"accuracy": accuracy, "precision": precision, "recall": recall}
    logging.info(f"\nMetrics for {approach}:")
    logging.info(f"  Accuracy:  {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall:    {recall:.4f}")

# --- Save Results ---
try:
    # Ensure the directory exists
    os.makedirs(os.path.dirname(EVALUATION_OUTPUT_CSV), exist_ok=True)

    # Save detailed results per example
    results_df.to_csv(EVALUATION_OUTPUT_CSV, index=False)
    logging.info(f"Detailed evaluation results saved to {EVALUATION_OUTPUT_CSV}")

    # Save summary metrics (optional, could print or save to another file)
    metrics_df = pd.DataFrame(metrics).T # Transpose for better view
    summary_path = EVALUATION_OUTPUT_CSV.replace(".csv", "_summary.csv")
    metrics_df.to_csv(summary_path)
    logging.info(f"Summary metrics saved to {summary_path}")

except Exception as e:
    logging.error(f"Failed to save evaluation results: {e}")

logging.info("Evaluation process completed.")