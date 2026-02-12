import os
import sys
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# CONFIGURATION
# Docker compose me service ka naam "ollama" hai, isliye http://ollama:11434 use karenge
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
VECTOR_DB_DIR = "./chroma_db_local"


def load_and_split_documents(folder_path: str = "data/docs"):
    all_chunks = []
    
    # Check if directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory {folder_path}. Please add PDFs here.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            print(f"Loading {file}...")

            loader = PyPDFLoader(path)
            documents = loader.load()

            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def build_vector_store(chunks=None):
    print(f"Connecting to Ollama at: {OLLAMA_BASE_URL}")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    if os.path.exists(VECTOR_DB_DIR) and chunks is None:
        return Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )


def build_rag_components(vector_store):
    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0.1,
        base_url=OLLAMA_BASE_URL
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 7}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Use the provided context to answer the question.
        If the answer is not present in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    return retriever, prompt, llm


def retrieve_context(retriever, question: str) -> Tuple[str, List]:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def generate_answer(prompt, llm, context: str, question: str) -> str:
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "question": question
    })


def initialize_rag():
    DOCS_FOLDER = "data/docs"

    if not os.path.exists(VECTOR_DB_DIR):
        print("Building vector database from documents...")
        chunks = load_and_split_documents(DOCS_FOLDER)
        if not chunks:
            print("No documents found or chunks created. Skipping vector store creation.")
            # Dummy return to prevent crash if empty
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vector_store = Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_DIR)
        else:
            vector_store = build_vector_store(chunks)
    else:
        print("Loading existing vector database...")
        vector_store = build_vector_store()

    retriever, prompt, llm = build_rag_components(vector_store)
    return retriever, prompt, llm