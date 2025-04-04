import os
import subprocess
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import csv
import google.generativeai as genai

# Prevent duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Environment variable set to prevent duplicate library error.")

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CodeBERT
codebert_model_name = "microsoft/codebert-base"
print(f"Loading CodeBERT tokenizer from {codebert_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(codebert_model_name)
print("Tokenizer loaded successfully.")
print(f"Loading CodeBERT model from {codebert_model_name} and moving it to {device}...")
codebert_model = AutoModel.from_pretrained(codebert_model_name).to(device)
print("CodeBERT model loaded and moved to device.")

# Configure Gemini API
api_key = "AIzaSyD-7OEjZ0lxxyU5Ukzvbdr9f2bJKZUhT9w"
if not api_key:
    print("Error: GEMINI_API_KEY not set. Exiting.")
    exit(1)
print("Gemini API key found. Configuring Gemini API...")
genai.configure(api_key=api_key)
print("Gemini API configured successfully.")

# Global FAISS index and file mapping
index = None
index_to_file = {}


def load_repo_links(file_path):
    """Load repository URLs from a file."""
    print(f"Loading repository links from {file_path}...")
    try:
        with open(file_path, "r") as file:
            repos = [line.strip() for line in file if line.strip()]
        print(f"Loaded {len(repos)} repository links.")
        return repos
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []


def clone_repos(repos):
    """Clone repositories locally."""
    print("Starting cloning process for repositories...")
    os.makedirs("repositories", exist_ok=True)
    for repo in repos:
        repo_name = repo.split("/")[-1].replace(".git", "")
        repo_path = os.path.join("repositories", repo_name)
        if not os.path.exists(repo_path):
            print(f"Cloning repository {repo} into {repo_path}...")
            try:
                subprocess.run(["git", "clone", repo, repo_path], check=True)
                print(f"Successfully cloned {repo_name}.")
            except subprocess.CalledProcessError as e:
                print(f"Error cloning {repo}: {e}")
        else:
            print(f"Repository {repo_name} already exists at {repo_path}.")
    print("Finished cloning repositories.")


def find_code_files(root_dir, extensions=[".py", ".java", ".c", ".cpp", ".js"]):
    """Find code files with specified extensions."""
    print(f"Scanning directory '{root_dir}' for code files with extensions: {extensions}")
    code_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                code_files.append(filepath)
    print(f"Found {len(code_files)} code files.")
    return code_files


def get_code_embedding(code_snippet):
    """Compute embedding using [CLS] token."""
    tokens = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = codebert_model(**tokens)
    embedding = output.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
    # Debug: print the shape of the computed embedding (should be (1, 768))
    print(f"Computed embedding of shape: {embedding.shape}")
    return embedding


def build_index(code_files, batch_size=32):
    """Build FAISS index with batch processing and cosine similarity."""
    global index, index_to_file
    print("Building FAISS index...")
    embedding_dim = 768
    index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
    embeddings = []
    total_files = len(code_files)
    for i in range(0, total_files, batch_size):
        batch_files = code_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: {len(batch_files)} files")
        batch_codes = []
        for file in batch_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    code = f.read()
                if code.strip():
                    batch_codes.append(code)
                else:
                    print(f"Skipped empty file: {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        if batch_codes:
            tokens = tokenizer(batch_codes, return_tensors="pt", truncation=True, padding=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                outputs = codebert_model(**tokens)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for j, emb in enumerate(batch_embeddings):
                emb = emb / np.linalg.norm(emb)  # Normalize for cosine similarity
                embeddings.append(emb)
                index_to_file[len(embeddings) - 1] = batch_files[j]
            print(f"Batch {i // batch_size + 1}: Processed {len(batch_embeddings)} embeddings.")
    if embeddings:
        embeddings = np.vstack(embeddings)
        index.add(embeddings)
        print(f"Total embeddings added to index: {embeddings.shape[0]}")
    else:
        print("No embeddings were generated.")
    print("FAISS index built successfully.")


def save_index(index, file_path):
    """Save FAISS index to disk."""
    print(f"Saving FAISS index to {file_path}...")
    faiss.write_index(index, file_path)
    print("FAISS index saved successfully.")


def find_similar_code(user_code, k=5):
    """Find top K similar code files using cosine similarity."""
    print("Searching for similar code files...")
    user_embedding = get_code_embedding(user_code)
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    distances, indices = index.search(user_embedding, k)
    matches = []
    for i in range(k):
        idx = indices[0][i]
        if idx != -1 and idx < len(index_to_file):
            matches.append((index_to_file[idx], float(distances[0][i])))
    print(f"Found {len(matches)} similar code file(s).")
    return matches


def check_plagiarism_with_gemini(user_code, similar_files=None):
    """Check plagiarism with Gemini, enforcing 'yes' or 'no' with references."""
    print("Checking plagiarism using Gemini API (Full System)...")
    prompt = (
            "You are a code plagiarism expert. Analyze the following user code snippet "
            "and determine if it is plagiarized. If similar code files are provided, "
            "compare them. Respond with exactly two words: 'yes' or 'no', followed by "
            "references (file paths) if 'yes', separated by commas. Example: 'yes, file1, file2' or 'no'.\n\n"
            "User code snippet:\n" + user_code + "\n\n"
    )
    if similar_files:
        prompt += "Similar code files:\n"
        for file_path, _ in similar_files[:3]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()[:1000]  # Truncate for LLM limits
                prompt += f"File: {file_path}\nCode:\n{code}\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().lower() if response and hasattr(response, "text") else ""
        print("Received response from Gemini API.")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return False, []  # Default to non-plagiarized on error

    parts = [p.strip() for p in response_text.split(",")]
    is_plagiarized = parts[0] == "yes"
    references = parts[1:] if is_plagiarized and len(parts) > 1 else []
    print(f"Gemini API determined plagiarism: {'yes' if is_plagiarized else 'no'}")
    if references:
        print(f"References from Gemini: {references}")
    return is_plagiarized, references


def check_plagiarism_rag_only(user_code, threshold=0.9):
    """RAG-only plagiarism check with cosine similarity."""
    print("Performing RAG-only plagiarism check...")
    matches = find_similar_code(user_code, k=5)
    plagiarized_files = [file for file, similarity in matches if
                         similarity > threshold]  # Cosine: higher is more similar
    print(f"RAG-only check found {len(plagiarized_files)} file(s) above the threshold of {threshold}.")
    return bool(plagiarized_files), plagiarized_files


def check_plagiarism_llm_only(user_code):
    """LLM-only plagiarism check."""
    print("Performing LLM-only plagiarism check...")
    prompt = (
            "You are a code plagiarism expert. Determine if the following code snippet "
            "is plagiarized. Respond with exactly two words: 'yes' or 'no'.\n\n"
            "User code snippet:\n" + user_code + "\n\n"
    )
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip().lower() if response and hasattr(response, "text") else ""
        print("Received response from Gemini API for LLM-only check.")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return False, []  # Default to non-plagiarized on error

    is_plagiarized = response_text.startswith("yes")
    print(f"LLM-only check determined plagiarism: {'yes' if is_plagiarized else 'no'}")
    return is_plagiarized, []


def evaluate_system():
    """Evaluate all three approaches and save to CSV."""
    print("Starting evaluation of plagiarism detection system...")
    test_cases = [
        {"id": 1, "code": "def add(a, b): return a + b", "ground_truth": "no"},
        {"id": 2, "code": "if (operation == 5) { cout << \"Result is: \" << sqrt(num1); }", "ground_truth": "yes"},
        {"id": 3, "code": "print('Hello, World!')", "ground_truth": "no"},
        {"id": 4, "code": "for i in range(10): print(i)", "ground_truth": "yes"},
    ]

    results = []
    for test in test_cases:
        print(f"Evaluating test case ID {test['id']}...")
        user_code = test["code"]
        # RAG only
        rag_result, rag_files = check_plagiarism_rag_only(user_code)
        results.append({
            "approach": "RAG_only",
            "snippet_id": test["id"],
            "prediction": "yes" if rag_result else "no",
            "ground_truth": test["ground_truth"],
            "references": ",".join(rag_files)
        })
        print(f"Test case {test['id']} RAG_only result: {'yes' if rag_result else 'no'}")
        # LLM only
        llm_result, llm_files = check_plagiarism_llm_only(user_code)
        results.append({
            "approach": "LLM_only",
            "snippet_id": test["id"],
            "prediction": "yes" if llm_result else "no",
            "ground_truth": test["ground_truth"],
            "references": ",".join(llm_files)
        })
        print(f"Test case {test['id']} LLM_only result: {'yes' if llm_result else 'no'}")
        # Full system
        similar_files = find_similar_code(user_code)
        full_result, full_files = check_plagiarism_with_gemini(user_code, similar_files)
        results.append({
            "approach": "full_system",
            "snippet_id": test["id"],
            "prediction": "yes" if full_result else "no",
            "ground_truth": test["ground_truth"],
            "references": ",".join(full_files)
        })
        print(f"Test case {test['id']} Full_system result: {'yes' if full_result else 'no'}")

    output_csv = "evaluation_results.csv"
    print(f"Saving evaluation results to {output_csv}...")
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["approach", "snippet_id", "prediction", "ground_truth", "references"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Evaluation results saved to evaluation_results.csv")


if __name__ == "__main__":
    # Step 1: Obtain repositories
    print("=== Step 1: Load repository links and clone repositories ===")
    repo_list = load_repo_links("repositories.txt")
    if not repo_list:
        print("No repositories specified. Create repositories.txt with GitHub URLs. Exiting.")
        exit(1)
    clone_repos(repo_list)

    # Step 2: Index code files
    print("\n=== Step 2: Find code files and build index ===")
    code_files = find_code_files("repositories")
    if not code_files:
        print("No code files found. Exiting.")
        exit(1)
    build_index(code_files)
    save_index(index, "code_embeddings.index")
    print("Vector database saved to code_embeddings.index")

    # Step 3: Evaluate system
    print("\n=== Step 3: Evaluate plagiarism detection system ===")
    evaluate_system()

    # Step 4: Test with sample code from first code file
    print("\n=== Step 4: Test plagiarism check on a sample code file ===")
    if code_files:
        sample_file = code_files[0]
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                user_code = f.read()
            print(f"Checking plagiarism for code from: {sample_file}")
            similar_files = find_similar_code(user_code)
            is_plagiarized, references = check_plagiarism_with_gemini(user_code, similar_files)
            print(f"Final plagiarism check result: {'Plagiarized' if is_plagiarized else 'Not plagiarized'}")
            if references:
                print("References provided by Gemini:")
                for ref in references:
                    print(f" - {ref}")
        except Exception as e:
            print(f"Error reading sample file {sample_file}: {e}")
    else:
        print("No sample code file available for testing.")
