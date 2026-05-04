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
Ingest three different documents

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A backup failure occurs when data cannot be written to storage or restored properly.",
    "doc_id": "doc1"
  }'

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A database transaction failure happens when ACID properties are violated during commit.",
    "doc_id": "doc2"
  }'

curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ollama is a local runtime for running large language models like Qwen and Mistral.",
    "doc_id": "doc3"
  }'


## retrieval
Use semantic query for retrieval process
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What happens when a backup fails?"
  }'