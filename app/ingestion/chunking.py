def chunk_text(text: str, size=800, overlap=120):
    """Chunk text into smaller pieces with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
