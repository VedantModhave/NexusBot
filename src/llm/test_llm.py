from openai import AsyncOpenAI
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your environment or .env file.")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    
    print("Testing Groq API connection...")
    try:
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=10,
            messages=[
                {"role": "system", "content": "You are a connection tester. Reply with 'CONNECTED' then stop."},
                {"role": "user", "content": "Hello!"}
            ]
        )
        print(f"Server response: {response.choices[0].message.content.strip()}")
        print("✅ SUCCESS: Groq API is working correctly.")
    except Exception as e:
        print(f"❌ FAILURE: Groq API error: {e}")

if __name__ == "__main__":
    asyncio.run(test_groq())
