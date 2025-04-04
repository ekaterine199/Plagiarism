# app/embedding_server/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# --- Add cache clearing ---
# Default Hugging Face cache directory
ARG HF_HOME=/root/.cache/huggingface
# Clear potential cache before installing anything that uses it
RUN rm -rf ${HF_HOME}/* && echo "Cleared Embedding Server HF Cache"
# --- End cache clearing ---

# Corrected paths relative to project root context
COPY app/embedding_server/requirements.txt .
# Use --no-cache-dir for pip as well, good practice
RUN pip install --no-cache-dir -r requirements.txt

# Corrected paths relative to project root context
COPY app/embedding_server/main.py .
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]