from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import config
import os

# 1. Load PDF
loader = PyPDFLoader(config.PDF_PATH)
documents = loader.load()  # list of Document objects

# 2. Split text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

# 4. Build FAISS vector store
vector_store = FAISS.from_documents(docs, embeddings)

# 5. Save vector store
if not os.path.exists("vector_db"):
    os.makedirs("vector_db")
vector_store.save_local(config.VECTOR_DB_PATH)

print("[INFO] Vector DB built from PDF and saved at", config.VECTOR_DB_PATH)
