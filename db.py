from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

DATA_PATH = r"data/pdf"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="LLM_based_testing")

loader = PyPDFDirectoryLoader(
    DATA_PATH,
    extract_images=False,
)

raw_data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    length_function = len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_data)

document = []
metadata = []
idx = []

i = 0
for x in chunks:
    document.append(x.page_content)
    idx.append("ID"+str(i))
    metadata.append(x.metadata)

    i += 1

collection.upsert(
    documents=document,
    metadatas=metadata,
    ids=idx
)