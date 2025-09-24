from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# Initialize FastAPI app
app = FastAPI()

# Load vector store
embeddings = OllamaEmbeddings(model="llama2")
vector_store = Chroma.load("vector_db", embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize chat model
chat_model = Ollama(model="llama2")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize RAG chain
chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Define request model
class Query(BaseModel):
    user_id: str
    question: str

# Define response model
class AnswerResponse(BaseModel):
    answer: str
    sources: list

# Define API endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask_medical(query: Query):
    result = chain({"question": query.question})
    sources = [doc.page_content[:200] for doc in result['source_documents']]
    return {"answer": result['answer'], "sources": sources}
