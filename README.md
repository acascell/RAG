# RAG
rag implementation retrieval ingestion

Architecture:

## LLM Runtime
- ollama 

## Chat models
- Small/Fast
  - qwen 2.5

- Ranking
  - mistral-small

## Embedding models
- nomic-embed-text

# Diagram
- User Question
   ↓
- Memory (last N messages)
   ↓
- Query Rewriter (LLM)
   ↓
- Embedding
   ↓
- Hybrid Retrieval
   ↓
- Reranker
   ↓
- Prompt (with context + history)
   ↓
- Streaming Answer

# Instructions
- docker compose build --no-cache
- docker compose up

## ingestion
Ingest three different documents to test the multiple cases

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
## Test 1
Use semantic query for retrieval process
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What happens when a backup fails?"
  }'
### expected behavior
A backup failure occurs when data cannot be written or restored properly.

## Test 2
test keyword-heavy
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ACID transaction commit failure database"
  }'

### expected behavior
database transaction failure ... ACID ...

## Test 3
Semantic + noisy query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why does my system not save data correctly when something breaks?"
  }'
