"""rag_pipeline_obs_openai.py - Complete RAG Pipeline with Langfuse Observability (OpenAI)

Required packages:
    pip install langfuse chromadb openai langchain langchain-community

Required env vars (.env):
    OPENAI_API_KEY=sk-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
"""

from langfuse import observe, get_client

# Drop-in replacement: same OpenAI SDK, but every call is traced automatically
# with model, token usage and calculated cost. Import path is the only change.
from langfuse.openai import OpenAI

from dotenv import load_dotenv
from typing import Dict, List
import chromadb
import os

# LangChain imports for document loading
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # dim 1536
EMBED_BATCH = 100  # texts per embeddings API call

# Initialize chromadb with persistent storage.
# NOTE: separate collection name from the old 384-dim MiniLM one — vectors of
# different dimensions cannot share a collection.
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="documents_openai", metadata={"hnsw:space": "cosine"}
)

# One client for the whole process — chat and embeddings both go through it,
# so both show up in the trace with their own cost.
oai = OpenAI()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts in one API call.

    Traced automatically by the langfuse.openai wrapper — no decorator needed.
    """
    response = oai.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        name="embed",
        metadata={"batch_size": len(texts)},
    )
    return [item.embedding for item in response.data]


# =============================================================================
# DOCUMENT INDEXING PIPELINE
# =============================================================================
@observe()
def load_and_index_documents(docs_dir: str = "./docs") -> int:

    documents = load_markdown_docs(docs_dir)
    if not documents:
        return 0

    chunks = chunk_documents(documents)
    return index_chunks(chunks)


@observe()
def load_markdown_docs(docs_dir: str) -> List:
    """Load all markdown files from directory using LangChain."""
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    # Use TextLoader for .md files (simpler, fewer dependencies)
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    # Update current span with metadata (Langfuse SDK v3)
    get_client().update_current_span(
        metadata={"docs_loaded": len(documents), "docs_dir": docs_dir}
    )
    return documents


@observe()
def chunk_documents(
    documents: List, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List:
    """Split documents into smaller chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    get_client().update_current_span(
        metadata={"total_chunks": len(chunks), "chunk_size": chunk_size}
    )
    return chunks


@observe()
def index_chunks(chunks: List) -> int:
    """Create embeddings and store in ChromaDB."""
    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk.page_content)
        metadatas.append({"source": chunk.metadata.get("source", "unknown")})

    # Batch the API calls: one request per 100 chunks, not one per chunk
    embeddings: List[List[float]] = []
    for start in range(0, len(documents), EMBED_BATCH):
        embeddings.extend(embed_texts(documents[start : start + EMBED_BATCH]))

    collection.upsert(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )

    get_client().update_current_span(
        metadata={
            "chunks_indexed": len(chunks),
            "embedding_model": EMBED_MODEL,
            "embed_api_calls": (len(documents) + EMBED_BATCH - 1) // EMBED_BATCH,
        }
    )
    return len(chunks)


# =============================================================================
# RAG QUERY PIPELINE
# =============================================================================


@observe()
def rag_pipeline(query: str) -> str:
    """
    Complete RAG pipeline with full observability.
    Each step becomes a span in the trace.
    """
    query_embedding = embed_query(query)
    chunks = retrieve_chunks(query_embedding)
    context = build_context(chunks)
    response = generate_response(query, context)
    return response


@observe()
def embed_query(query: str) -> List[float]:
    """Embed the user query."""
    embedding = embed_texts([query])[0]

    get_client().update_current_span(
        metadata={
            "query_length": len(query),
            "embedding_dim": len(embedding),
            "embedding_model": EMBED_MODEL,
        }
    )
    return embedding


@observe()
def retrieve_chunks(embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Retrieve relevant document chunks from ChromaDB."""
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            chunks.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

    get_client().update_current_span(
        metadata={
            "chunks_retrieved": len(chunks),
            "avg_distance": (
                sum(c["distance"] for c in chunks) / len(chunks) if chunks else 0
            ),
        }
    )
    return chunks


@observe()
def build_context(chunks: List[Dict]) -> str:
    """Assemble retrieved chunks into context string."""
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(f"[Source {i+1} - {source}]: {chunk['content']}")

    context = "\n\n".join(context_parts)

    get_client().update_current_span(
        metadata={"context_length": len(context), "num_chunks_used": len(chunks)}
    )
    return context


@observe()
def generate_response(query: str, context: str) -> str:
    """Generate response with OpenAI.

    No manual usage logging here: the langfuse.openai wrapper creates the
    GENERATION observation itself, with model, input/output tokens and the
    calculated cost. This function stays a plain SPAN wrapping that call.
    """
    system_prompt = (
        "Answer the question using only the provided context. "
        "If the context doesn't contain relevant information, say so."
    )

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = oai.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # These kwargs are consumed by Langfuse, not forwarded to OpenAI
        name="rag-answer",
        metadata={"context_length": len(context), "step": "generation"},
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Step 1: Index documents (run once or when docs change)
    print("Indexing documents from ./docs folder...")
    try:
        num_indexed = load_and_index_documents("./docs")
        print(f"Indexed {num_indexed} chunks")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Create a 'docs' folder with .md files first.")
        exit(1)

    # Step 2: Query the RAG pipeline
    print("\nQuerying RAG pipeline...")
    result = rag_pipeline("What are the main components of langgraph?")
    print(f"\nResponse:\n{result}")

    # Always flush in short-lived scripts
    get_client().flush()