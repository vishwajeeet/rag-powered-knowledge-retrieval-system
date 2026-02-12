from fastapi import FastAPI
from app.rag_core import initialize_rag, retrieve_context, generate_answer
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

app = FastAPI()

retriever = None
prompt = None
llm = None


@app.on_event("startup")
def startup_event():
    global retriever, prompt, llm
    print("Initializing RAG system...")
    retriever, prompt, llm = initialize_rag()
    print("RAG system ready.")


@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    context, _ = retrieve_context(retriever, request.question)
    answer = generate_answer(prompt, llm, context, request.question)
    return {"answer": answer}
