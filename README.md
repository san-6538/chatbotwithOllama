A Retrieval-Augmented Generation (RAG) chatbot for medical queries using a PDF medical encyclopedia as context.
Powered by Ollama LLM and LangChain, this chatbot provides answers along with relevant source snippets.

Features

Loads a PDF medical encyclopedia for context.

Splits PDF text into chunks and stores embeddings in a Chroma vector database.

Uses RAG approach to retrieve relevant context before generating answers.

Maintains user-specific chat history using LangChain memory.

Returns answers and top source snippets for verification.

Fully local deployment using Ollama, no external API calls needed.
