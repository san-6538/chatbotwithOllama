import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# Load PDF
reader = PdfReader("data/medical_encyclopedia.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)

# Generate embeddings
embeddings = OllamaEmbeddings(model="llama2")

# Create Chroma vector store
vector_store = Chroma.from_documents(chunks, embeddings)

# Save vector store
vector_store.persist()

