from openai import AsyncOpenAI
import os

RESPONSE_SYSTEM_PROMPT = """
You are NexusBot, a helpful and friendly college assistant for XYZ College.

RULES (follow strictly):
1. Answer ONLY using the provided CONTEXT. Do not use outside knowledge.
2. Preserve ALL numbers, ₹ values, percentages, URLs, dates EXACTLY as 
   they appear in the context. Never round, omit, or paraphrase numbers.
3. Be concise — 2 to 4 sentences maximum.
4. Sound natural and conversational, not like a document dump.
5. If the context does not answer the question, say exactly:
   "I don't have specific information about that. Please contact 
    info@xyzcollege.edu.in for assistance."
6. NEVER mention that you are using a "context" or "chunk" or "document".
7. Respond in the SAME LANGUAGE as specified in the user language field.
   - lang=en  → English
   - lang=hi  → Hindi (Devanagari script)
   - lang=mr  → Marathi (Devanagari script)
8. Do not add greetings or sign-offs unless the user greeted you.
"""

async def generate_llm_response(
    query: str,
    context_chunk: str,
    lang: str,
    client: AsyncOpenAI
) -> str:
    """Generate a conversational response from a context chunk in the target language."""
    user_message = f"""
User language: {lang}
User question: {query}

CONTEXT:
{context_chunk}

Answer the question using only the context above.
"""
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=[
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] response generation: {e}")
        return "I'm sorry, I'm having trouble generating a response. Please try again or contact support."
