def chunk_text(text: str, size=800, overlap=120):
    """Split text into overlapping chunks for embedding and indexing.

    Uses a sliding window approach to break long text into smaller segments
    that fit within embedding model context limits while preserving continuity
    between adjacent chunks.

    Args:
        text: The input text to split into chunks.
        size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks,
            ensuring context is preserved at chunk boundaries.

    Returns:
        A list of text chunks.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
