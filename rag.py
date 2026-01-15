from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

pdf_path = r"data\pdf\Amith_Benny_Resume.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(chunks, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

question = "what is llm?"
relevant_docs = retriever.invoke(question)

context = "\n\n".join(
    f"(Page {doc.metadata.get('page', 'N/A')}) {doc.page_content}"
    for doc in relevant_docs
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Answer ONLY using the given context."
    },
    {
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}
"""
    }
]

client = InferenceClient(token=HF_TOKEN)

response = client.chat_completion(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=messages,
    max_tokens=512,
    temperature=0.2
)

answer = response.choices[0].message.content
print(answer)