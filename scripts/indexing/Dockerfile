# scripts/indexing/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# --- Add cache clearing ---
# Default Hugging Face cache directory
ARG HF_HOME=/root/.cache/huggingface
# Clear potential cache before installing anything that uses it
RUN rm -rf ${HF_HOME}/* && echo "Cleared Indexing HF Cache"
# --- End cache clearing ---

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Corrected paths relative to project root context
COPY scripts/indexing/requirements.txt .
# Use --no-cache-dir for pip
RUN pip install --no-cache-dir -r requirements.txt

# Corrected paths relative to project root context
COPY scripts/indexing/indexer.py .
COPY config.yaml .

CMD ["python", "indexer.py"]