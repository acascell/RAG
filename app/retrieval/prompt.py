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

    prompt = f"""
    You are a precise assistant that answers questions based strictly on the provided context.

    RULES:
    - Use ONLY facts explicitly stated in the Context below.
    - Do NOT add any information from your own knowledge.
    - Keep your answer short and directly quote or paraphrase the Context.
    - If the Context does not contain the answer, respond with: "I don't have enough information to answer this question."
    
    Conversation History:
    {history_text}
    
    Context:
    {ctx}
    
    Question:
    {question}
    
    Answer:"""

    # Debug: Print the prompt to see what's being sent to the model
    #print(f"\n[DEBUG PROMPT]\nQuestion: {question}\nContext: {ctx}\n[END DEBUG PROMPT]\n")

    return prompt
