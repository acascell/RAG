def build_prompt(question: str, contexts: list[str], history: list[dict]):

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
