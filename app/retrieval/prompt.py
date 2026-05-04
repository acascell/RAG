def build_prompt(question: str, contexts: list[str]):
    ctx = "\n\n".join(contexts)

    return f"""
    You are a precise assistant.
    Answer the questions using ONLY the context defined below.
    Context: 
    {ctx}

    Question:
    {question}

    Answer:
    """
