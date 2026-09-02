def rag_answer_prompt(question: str, context_chunks: list[str]) -> list[dict]:
    context = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are a study assistant answering questions strictly from the provided "
        "document excerpts. Only use information present in the context. If the "
        "context doesn't fully answer the question, say what's missing rather than "
        "inventing details. Use Markdown."
    )
    user = f"Context excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def hallucination_score_prompt(question: str, answer: str, context_chunks: list[str]) -> list[dict]:
    context = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are a strict fact-checker. Respond only with valid JSON, no prose. "
        "Score how well the ANSWER is supported by the CONTEXT, from 0 (pure "
        "hallucination, unsupported by context) to 100 (fully grounded in context)."
    )
    user = (
        f"Context:\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer: {answer}\n\n"
        'Respond as JSON: {"score": <0-100 integer>}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
