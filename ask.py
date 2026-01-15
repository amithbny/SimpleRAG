import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="LLM_based_testing"
)

user_query = input(
    "Hello there, what do you wanna know? \n\n"
)

results = collection.query(
    query_texts=[user_query],
    n_results=1
)

context = " ".join(results["documents"][0])

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

system_prompt = f"""
You are a helpful assistant. You answer questions about software testing using
Large Language Models.

RULES:
# - Answer ONLY using the provided context
# - Do NOT use outside knowledge
- Do NOT hallucinate
- If the answer is not in the context, say: "I don't know"

--------------------
Context:
{context}
"""

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ],
    temperature=0.2,
    max_tokens=300
)

print("\n---------------------\n")
print(response.choices[0].message.content)
