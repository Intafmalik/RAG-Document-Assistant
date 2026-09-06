import click
import sys
from typing import List, Dict, Any, Optional

from .document_loader import RamayanaPDFLoader
from .chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    StructuralChunker,
)
from .vector_store import VectorStoreManager
from .retrieval.hybrid import HybridRetriever
from .retrieval.rerank import CrossEncoderReranker
from .qa_pipeline import RAGPipeline, RAGAnswer
from . import config


@click.group()
def cli():
    """RAG Application for Ramayana document queries."""
    pass


@cli.command()
@click.option("--strategy", type=click.Choice(["fixed", "recursive", "semantic", "structural"]),
              default="recursive", help="Chunking strategy to use")
@click.option("--rebuild/--no-rebuild", default=True,
              help="Rebuild the vector store index")
def build(strategy: str, rebuild: bool):
    """Build the vector store index from the Ramayana PDF."""
    click.echo(f"[{click.style('INFO', fg='cyan')}] Loading Ramayana PDF...")

    # Load PDF
    loader = RamayanaPDFLoader(config.PDF_PATH)
    try:
        docs = loader.load()
        click.echo(f"[{click.style('INFO', fg='cyan')}] Extracted {len(docs)} pages from PDF")
    except Exception as e:
        click.echo(f"[{click.style('ERROR', fg='red')}] Failed to load PDF: {str(e)}")
        sys.exit(1)

    # Extract full text for corpus
    full_text = loader.load_all_text()
    click.echo(f"[{click.style('INFO', fg='cyan')}] Full text length: {len(full_text)} characters")

    # Initialize embedding function (sentence-transformers)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(config.CHUNKING_STRATEGIES["semantic"]["embedding_model"])

    # Chunk using the selected strategy
    click.echo(f"[{click.style('INFO', fg='cyan')}] Chunking with {strategy} strategy...")

    chunker_map = {
        "fixed": FixedSizeChunker(),
        "recursive": RecursiveChunker(),
        "semantic": SemanticChunker(),
        "structural": StructuralChunker(),
    }

    chunker = chunker_map[strategy]
    all_chunks = []

    # Process pages
    page_docs = docs  # Already page-split
    for page_doc in page_docs:
        page_text = page_doc.page_content
        page_meta = {**page_doc.metadata, "page": page_doc.metadata.get("page", 0)}
        chunks = chunker.chunk(page_text, page_meta)
        all_chunks.extend(chunks)

    click.echo(f"[{click.style('INFO', fg='cyan')}] Generated {len(all_chunks)} chunks")

    # Initialize vector store
    vs_manager = VectorStoreManager(config.CHROMA_PERSIST_DIR)

    if rebuild:
        # Delete existing collection for this strategy
        vs_manager.delete_collection(strategy)

    # Add documents to vector store
    vs_manager.add_documents(
        strategy_name=strategy,
        documents=all_chunks,
        embedding_function=embedder.encode,
    )

    # Also prepare BM25 corpus for hybrid search
    corpus = [chunk.page_content for chunk in all_chunks]

    click.echo(
        f"[{click.style('SUCCESS', fg='green')}] Build complete! "
        f"Strategy: {strategy}, Chunks: {len(all_chunks)}"
    )


@cli.command()
@click.argument("question", required=False)
@click.option("--strategy", type=click.Choice(["fixed", "recursive", "semantic", "structural"]),
              default="recursive", help="Chunking strategy to query")
@click.option("--top-k", default=5, help="Number of sources to retrieve")
@click.option("--enable-tools", is_flag=True, help="Enable tool calling for the LLM")
@click.option("--model", default="gemini-primary", help="LiteLLM model name to use")
def ask(question: str, strategy: str, top_k: int, enable_tools: bool, model: str):
    """Ask a question about the Ramayana document."""
    if not question:
        click.echo("Please provide a question.")
        click.echo("Usage: python -m rag_app ask 'Your question here'")
        sys.exit(1)

    click.echo(f"[{click.style('INFO', fg='cyan')}] Using strategy: {strategy}")
    click.echo(f"[{click.style('INFO', fg='cyan')}] Question: {question}")

    # Initialize vector store
    vs_manager = VectorStoreManager(config.CHROMA_PERSIST_DIR)

    # Initialize embedding function
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(config.CHUNKING_STRATEGIES["semantic"]["embedding_model"])

    # Initialize hybrid retriever - we need the BM25 corpus
    # Load chunks from the strategy collection
    collection = vs_manager.get_collection(strategy)
    count = collection.count()

    if count == 0:
        click.echo(
            f"[{click.style('ERROR', fg='red')}] No documents found. "
            f"Run 'python -m rag_app build --strategy {strategy}' first."
        )
        sys.exit(1)

    # Initialize components
    from .document_loader import RamayanaPDFLoader
    loader = RamayanaPDFLoader(config.PDF_PATH)
    full_text = loader.load_all_text()

    # For BM25, we need the corpus - use existing chunks' text
    # In a full implementation, this would be stored/loaded persistently
    # For now, we'll re-chunk to get the corpus
    chunker = RecursiveChunker()  # Default for BM25 corpus
    # Actually, let's just use the full text split simply for BM25 demo
    import re
    sentences = re.split(r"[.!?]+", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunk_size = 200
    bm25_corpus = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:min(i+chunk_size, len(sentences))]) + "."
        bm25_corpus.append(chunk)

    hybrid_retriever = HybridRetriever(
        vector_store=vs_manager,
        bm25_corpus=bm25_corpus,
        embedding_function=embedder.encode,
        k_dense=config.DEFAULT_RETRIEVAL_K,
        k_sparse=config.DEFAULT_RETRIEVAL_K,
        rrf_k=config.RRF_K,
    )

    # Initialize reranker
    reranker = CrossEncoderReranker()

    # Initialize LLM via LiteLLM proxy
    from openai import OpenAI
    # Note: In production, use the ChatLiteLLM from qa_pipeline
    # Here we directly use OpenAI client for simplicity in CLI

    # Initialize QA pipeline
    pipeline = RAGPipeline(
        vector_store=vs_manager,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_model=model,
        strategy=strategy,
        enable_tools=enable_tools,
    )

    # Ask the question
    click.echo(f"[{click.style('INFO', fg='cyan')}] Retrieving and generating answer...")

    rag_answer = pipeline.ask(question, top_k=top_k)

    # Display structured answer
    click.echo(f"\n{click.style('Answer:', fg='green')} {rag_answer.answer}")
    click.echo(f"\n{click.style('Confidence:', fg='green')} {rag_answer.confidence:.2f}")

    if rag_answer.sources:
        click.echo(f"\n{click.style('Sources:', fg='cyan')}")
        for i, source in enumerate(rag_answer.sources, 1):
            click.echo(f"  {i}. {source}")

    if rag_answer.reasoning:
        click.echo(f"\n{click.style('Reasoning:', fg='yellow')} {rag_answer.reasoning}")


@cli.command()
def compare():
    """Compare all four chunking strategies."""
    click.echo("Comparing chunking strategies for Ramayana document...")
    click.echo("=" * 60)

    strategies = ["fixed", "recursive", "semantic", "structural"]

    for strategy in strategies:
        click.echo(f"\n{click.style(f'Strategy: {strategy}', fg='cyan', bold=True)}")

        # Check if collection exists
        from .vector_store import VectorStoreManager
        vs_manager = VectorStoreManager(config.CHROMA_PERSIST_DIR)
        collection = vs_manager.get_collection(strategy)
        count = collection.count()

        if count == 0:
            click.echo(f"  No documents indexed. Run 'python -m rag_app build --strategy {strategy}' first.")
            continue

        # Simple retrieval test
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(config.CHUNKING_STRATEGIES["semantic"]["embedding_model"])

        # Test a few questions
        test_questions = [
            "Who is Rama?",
            "What is the story of Sita?",
        ]

        total_chunks = 0
        for q in test_questions:
            results = vs_manager.similarity_search(
                strategy_name=strategy,
                query=q,
                embedding_function=embedder.encode,
                k=3,
            )
            total_chunks += len(results)

        click.echo(f"  Total chunks: {count}")
        click.echo(f"  Test retrieval: {len(test_questions)} questions, {total_chunks} total results")


@cli.command()
def stats():
    """Show vector store statistics."""
    vs_manager = VectorStoreManager(config.CHROMA_PERSIST_DIR)
    stats = vs_manager.get_collection_stats()

    click.echo("Vector Store Statistics:")
    click.echo("=" * 60)
    for name, info in stats.items():
        count = info.get("document_count", 0)
        meta = info.get("metadata", {})
        click.echo(f"  {name}: {count} documents")
        if meta:
            click.echo(f"    Strategy: {meta.get('strategy', 'N/A')}")


if __name__ == "__main__":
    cli()