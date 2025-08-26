SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer the user's question.
- Cite sources with [source] where source is the origin (filename, URL title, or page).
- If the answer is not in the context, say you don't know based on the provided documents.
- Be concise and factual.
"""

USER_TEMPLATE = """Question: {question}

Context:
{context}

Instructions:
- Use only the context to answer.
- Add citations like [{{source}}] inline where relevant.
- If context is insufficient, say so and suggest where to look next.

Answer:"""

def build_prompt(question: str, contexts: list) -> str:
    compiled = []
    for c in contexts:
        compiled.append(f"Source: {c['source']}\nSnippet: {c['text']}")
    
    ctx_block = "\n\n".join(compiled) if compiled else "(no relevant context)"
    return USER_TEMPLATE.format(question=question, context=ctx_block)
