import os
import subprocess
import json
import yaml
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
CONFIG_PATH = "/app/config.yaml"
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully.")
except FileNotFoundError:
    logging.error(f"Configuration file not found at {CONFIG_PATH}")
    exit(1)
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    exit(1)

REPO_URLS = config.get('repositories', [])
REPO_CLONE_DIR = config.get('repo_clone_dir', 'data/cloned_repos')
CODE_EXTENSIONS = config.get('code_extensions', ['.py'])
EMBEDDING_MODEL = config.get('embedding_model', 'microsoft/codebert-base')
FAISS_INDEX_PATH = config.get('faiss_index_path', 'data/faiss_index/code_index.faiss')
FAISS_METADATA_PATH = config.get('faiss_metadata_path', 'data/faiss_index/code_metadata.json')

# Ensure parent directories exist
os.makedirs(REPO_CLONE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

# --- Device Selection (GPU if available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- Model Loading ---
try:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL, from_tf=True).to(device)
    logging.info(f"Embedding model '{EMBEDDING_MODEL}' loaded.")
except Exception as e:
    logging.error(f"Failed to load model or tokenizer: {e}")
    exit(1)

# --- Helper Functions ---
def clone_repo(repo_url, target_dir):
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_path = os.path.join(target_dir, repo_name)
    if os.path.exists(clone_path):
        logging.info(f"Repository {repo_name} already cloned. Skipping.")
        return clone_path
    try:
        logging.info(f"Cloning {repo_url} into {clone_path}...")
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, clone_path], check=True, capture_output=True)
        logging.info(f"Successfully cloned {repo_name}.")
        return clone_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone {repo_url}: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during cloning {repo_url}: {e}")
        return None

def get_code_files(repo_path, extensions):
    code_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                code_files.append(os.path.join(root, file))
    return code_files

def get_embedding(code_text):
    try:
        # Truncate long code to avoid exceeding model limits
        inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token embedding or mean pooling
        # Using mean pooling here for potentially better representation of the whole snippet
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        logging.warning(f"Could not generate embedding: {e}")
        return None

# --- Main Indexing Logic ---
all_embeddings = []
all_metadata = [] # Store file paths corresponding to embeddings

logging.info("Starting repository cloning...")
cloned_repo_paths = [clone_repo(url, REPO_CLONE_DIR) for url in REPO_URLS]
cloned_repo_paths = [path for path in cloned_repo_paths if path is not None] # Filter out failed clones

if not cloned_repo_paths:
    logging.error("No repositories were successfully cloned. Exiting.")
    exit(1)

logging.info("Starting code file indexing...")
file_count = 0
for repo_path in cloned_repo_paths:
    logging.info(f"Scanning repository: {repo_path}")
    code_files = get_code_files(repo_path, CODE_EXTENSIONS)
    logging.info(f"Found {len(code_files)} code files in {os.path.basename(repo_path)}.")

    for file_path in code_files:
        relative_path = os.path.relpath(file_path, REPO_CLONE_DIR) # Store relative path
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            if not code.strip(): # Skip empty files
                logging.warning(f"Skipping empty file: {relative_path}")
                continue

            embedding = get_embedding(code)

            if embedding is not None and embedding.shape: # Check if embedding is valid
                all_embeddings.append(embedding)
                all_metadata.append({"file_path": relative_path}) # Store metadata
                file_count += 1
                if file_count % 50 == 0:
                    logging.info(f"Processed {file_count} files...")

        except FileNotFoundError:
            logging.warning(f"File not found during processing: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {relative_path}: {e}")

if not all_embeddings:
    logging.error("No embeddings were generated. Check logs for errors.")
    exit(1)

logging.info(f"Generated {len(all_embeddings)} embeddings.")

# --- Faiss Indexing ---
embeddings_np = np.array(all_embeddings).astype('float32')
faiss.normalize_L2(embeddings_np)
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)

# index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean)
# For cosine similarity, use IndexFlatIP and normalize vectors before adding/searching
# faiss.normalize_L2(embeddings_np)
# index = faiss.IndexFlatIP(dimension)

index.add(embeddings_np)
logging.info(f"Faiss index created with {index.ntotal} vectors.")

# --- Save Index and Metadata ---
try:
    faiss.write_index(index, FAISS_INDEX_PATH)
    logging.info(f"Faiss index saved to {FAISS_INDEX_PATH}")

    with open(FAISS_METADATA_PATH, 'w') as f:
        json.dump(all_metadata, f, indent=4)
    logging.info(f"Metadata saved to {FAISS_METADATA_PATH}")

except Exception as e:
    logging.error(f"Failed to save Faiss index or metadata: {e}")
    exit(1)

logging.info("Indexing process completed successfully.")