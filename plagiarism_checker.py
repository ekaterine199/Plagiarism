import os
import subprocess
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import csv
import google.generativeai as genai
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


def chunk_code(code, max_tokens=500):
    """Split code into chunks of max_tokens."""
    lines = code.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(tokenizer.tokenize(line))
        if current_length + line_length > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks


def get_code_embeddings_batch(code_snippets):
    """Compute embeddings for a batch of code snippets."""
    tokens = tokenizer(code_snippets, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = codebert_model(**tokens)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
    return embeddings


def build_index(code_files, batch_size=32):
    """Build FAISS index with batch processing, grouped by file extensions."""
    global index, index_to_file
    print("Building FAISS index...")
    embedding_dim = 768
    index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
    embeddings = []
    total_files = len(code_files)

    # Group files by extension
    extension_groups = {}
    for file in code_files:
        ext = os.path.splitext(file)[1]
        if ext not in extension_groups:
            extension_groups[ext] = []
        extension_groups[ext].append(file)

    # Process each group
    for ext, files in extension_groups.items():
        print(f"Processing {len(files)} files with extension {ext}...")
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_codes = []
            file_to_chunks = {}  # Map each file to its chunks
            for file in batch_files:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        code = f.read()
                    if code.strip():
                        chunks = chunk_code(code)  # Split large files into chunks
                        batch_codes.extend(chunks)
                        file_to_chunks[file] = len(chunks)  # Store number of chunks for this file
                    else:
                        print(f"Skipped empty file: {file}")
                except Exception as e:
                    print(f"Error reading {file}: {e}")

            if batch_codes:
                batch_embeddings = get_code_embeddings_batch(batch_codes)  # Batch processing
                embedding_index = 0
                for file, num_chunks in file_to_chunks.items():
                    for _ in range(num_chunks):
                        embeddings.append(batch_embeddings[embedding_index])
                        index_to_file[len(embeddings) - 1] = (ext, file)  # Map to original file
                        embedding_index += 1
                print(f"Processed {len(batch_embeddings)} embeddings for {ext} files.")

    if embeddings:
        embeddings = np.vstack(embeddings)
        index.add(embeddings)
        print(f"Total embeddings added to index: {embeddings.shape[0]}")
    else:
        print("No embeddings were generated.")
    print("FAISS index built successfully.")

    # Visualize embeddings after building the index
    labels = [index_to_file[i][1] for i in range(len(embeddings))]  # Use file paths as labels
    visualize_embeddings(embeddings, labels, method="tsne", title="Code Embeddings Visualization")


def visualize_embeddings(embeddings, labels, method="tsne", title="Embedding Visualization"):
    """
    Visualize embeddings using T-SNE or PCA.
    
    Args:
        embeddings (np.array): Array of embeddings (n_samples, n_features).
        labels (list): List of labels for each embedding (e.g., file paths or extensions).
        method (str): "tsne" or "pca".
        title (str): Title of the plot.
    """
    print(f"Visualizing embeddings using {method.upper()}...")
    
    if method == "tsne":
        # T-SNE for 2D visualization
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(1, n_samples - 1))  # Ensure perplexity < n_samples
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)
    elif method == "pca":
        # # PCA for 2D visualization
        # pca = PCA(n_components=2, random_state=42)
        # reduced_embeddings = pca.fit_transform(embeddings)
        n_samples, n_features = embeddings.shape
        n_components = min(2, n_samples, n_features)  # Ensure valid n_components
        
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'.")
    
    # Plot the reduced embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=np.arange(len(labels)), cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Sample Index")
    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    
    # Annotate a few points for clarity
    for i, label in enumerate(labels[:10]):  # Annotate first 10 points
        plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)
    
    # Save plot in the same folder as the script
    output_path = os.path.join(os.getcwd(), "code_embeddings_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Visualization saved as {output_path}")


def save_index(index, file_path):
    """Save FAISS index to disk."""
    print(f"Saving FAISS index to {file_path}...")
    faiss.write_index(index, file_path)
    print("FAISS index saved successfully.")


def find_similar_code(user_code, k=5):
    """Find top K similar code files using cosine similarity."""
    print("Searching for similar code files...")
    user_embedding = get_code_embeddings_batch([user_code])[0]  # Single embedding
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    distances, indices = index.search(np.array([user_embedding]), k)
    matches = []
    for i in range(k):
        idx = indices[0][i]
        if idx != -1 and idx < len(index_to_file):
            ext, file_path = index_to_file[idx]  # Extract extension and file path
            matches.append((file_path, float(distances[0][i]), ext))  # Include extension in the result
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
        for file_path, _, ext in similar_files[:3]:  # Unpack file_path, similarity, and extension
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()[:1000]  # Truncate for LLM limits
                prompt += f"File: {file_path} (Extension: {ext})\nCode:\n{code}\n\n"  # Include extension in the prompt
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
    plagiarized_files = [file for file, similarity, ext in matches if similarity > threshold]  # Unpack file_path, similarity, and extension
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