def build_prompt(question: str, contexts: list[str], history: list[dict]):
    """Construct the final LLM prompt combining conversation history, retrieved context, and the question.

    Args:
        question: The rewritten user query.
        contexts: List of relevant text passages retrieved from the index.
        history: List of conversation turns, each a dict with 'role' and 'content' keys.

    Returns:
        A formatted prompt string ready to be sent to the LLM.
    """

    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history])

    ctx = "\n\n".join(contexts)

    return f"""
    You are a precise assistant.
    
    Conversation:
    {history_text}
    
    Context: 
    {ctx}

    Question:
    {question}

    Answer clearly and concisely.
    """
