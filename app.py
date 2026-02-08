import os
import sys
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
VECTOR_DB_DIR = "./chroma_db_local"


def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks=None):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

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
        temperature=0.1
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


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    if not os.path.exists(pdf_path):
        print("PDF file not found.")
        sys.exit(1)

    if not os.path.exists(VECTOR_DB_DIR):
        chunks = load_and_split_pdf(pdf_path)
        vector_store = build_vector_store(chunks)
    else:
        vector_store = build_vector_store()

    retriever, prompt, llm = build_rag_components(vector_store)

    question = "What is the main topic of this document?"

    context, retrieved_docs = retrieve_context(retriever, question)
    answer = generate_answer(prompt, llm, context, question)

    print("Answer:")
    print(answer)
