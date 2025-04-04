# app/plagiarism_checker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Corrected paths relative to project root context
COPY app/plagiarism_checker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Corrected paths relative to project root context
COPY app/plagiarism_checker/main.py .
COPY config.yaml .

EXPOSE 8001

ENV EMBEDDING_SERVER_URL="http://embedding-server:8000/embed"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]