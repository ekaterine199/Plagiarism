# scripts/evaluation/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Corrected paths relative to project root context
COPY scripts/evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Corrected paths relative to project root context
COPY scripts/evaluation/evaluator.py .
COPY scripts/evaluation/test_data.json .
COPY config.yaml .

CMD ["python", "evaluator.py"]