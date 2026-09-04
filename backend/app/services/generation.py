from ollama import chat


MODEL_NAME = "qwen3:1.7b"


def generate_answer(question: str, retrieved_chunks: list):
    """
    Generate a grounded answer using retrieved document chunks.
    """

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are a question-answering assistant for
The Complete Sherlock Holmes by Arthur Conan Doyle.

Answer the user's question using ONLY the provided context.

If the answer cannot be determined from the context, say:
"I could not find enough information in the provided context."

Do not use outside knowledge.
Do not invent facts.

Always mention the relevant source page(s).

Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]