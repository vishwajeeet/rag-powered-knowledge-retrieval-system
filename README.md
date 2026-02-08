# Local RAG System using Ollama

This project implements a fully local Retrieval-Augmented Generation (RAG)
system to answer questions from PDF documents using open-source Large
Language Models.

The system focuses on reducing hallucination by grounding responses in
retrieved document context while remaining cost-efficient and privacy-friendly.

---

## Architecture Overview

1. PDF ingestion and parsing
2. Recursive text chunking for semantic coherence
3. Vector embedding generation using Ollama
4. Vector storage and similarity search with ChromaDB
5. Context-aware answer generation using a local LLM

---

## Design Decisions

- Optimized for correctness and explainability rather than scale
- Single-document assumption to keep retrieval deterministic
- Explicit separation between retrieval and generation stages
- Low-temperature generation with fallback to "I don't know" to limit hallucination

---

## Usage

1. Place the target PDF in the project directory
2. Run the script to generate vector embeddings
3. Ask questions via the command line

### Changing the source document

Vector embeddings are document-specific.  
If a different PDF is used, delete the existing vector database and re-run
the script to rebuild embeddings.

```bash
rm -rf chroma_db_local
```

