# RAG
rag implementation retrieval ingestion

Architecture:

## LLM Runtime
- ollama 

## Chat models
- Small/Fast
  - qwen 2.5

- Better reasoning
  - mistral-small

## Embedding models
- nomic-embed-text

# Diagram
query
  ↓
embed query
  ↓
vector search (OpenSearch kNN)
  ↓
BM25 search (OpenSearch text)
  ↓
merge results
  ↓
rerank (LLM)
  ↓
prompt builder
  ↓
Ollama generation

# Instructions
docker compose build --no-cache
docker compose up

## ingestion
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A backup failure occurs when data cannot be stored or restored correctly.",
    "doc_id": "doc1"
  }'


## retrieval
curl -X POST http://localhost:8000/ask \   
  -H "Content-Type: application/json" \
  -d '{"question": "What is a backup failure?", "model": "qwen2.5"}'