"""Generate sample PDFs for testing the Document Analysis system."""

import fitz  # PyMuPDF
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent / "sample-pdfs"
SAMPLE_DIR.mkdir(exist_ok=True)


def create_pdf(filename: str, pages: list[dict]):
    """Create a PDF with the given pages. Each page has 'title' and 'body'."""
    doc = fitz.open()
    for page_data in pages:
        page = doc.new_page(width=612, height=792)  # Letter size
        # Title
        page.insert_text(
            (72, 72),
            page_data["title"],
            fontsize=18,
            fontname="helv",
            color=(0.1, 0.1, 0.4),
        )
        # Body text — wrap manually
        y = 110
        for line in page_data["body"].split("\n"):
            if y > 740:
                break
            page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 16

    doc.save(str(SAMPLE_DIR / filename))
    doc.close()
    print(f"Created: {filename} ({len(pages)} pages)")


# ── Sample 1: Machine Learning Basics ────────────────────────────────
create_pdf("machine-learning-basics.pdf", [
    {
        "title": "Introduction to Machine Learning",
        "body": """Machine learning is a subset of artificial intelligence that focuses on building
systems that learn from data. Instead of being explicitly programmed, these
systems improve their performance on a specific task through experience.

There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data,
   making predictions based on input-output pairs. Common algorithms include
   linear regression, decision trees, and neural networks.

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
   Clustering (K-means, DBSCAN) and dimensionality reduction (PCA, t-SNE)
   are common techniques.

3. Reinforcement Learning: An agent learns to make decisions by interacting
   with an environment, receiving rewards or penalties for its actions.

Key concepts in machine learning include:
- Features: The input variables used to make predictions
- Labels: The output variable we want to predict
- Training set: Data used to train the model
- Test set: Data used to evaluate the model
- Overfitting: When a model performs well on training data but poorly on new data
- Underfitting: When a model is too simple to capture the underlying patterns""",
    },
    {
        "title": "Neural Networks and Deep Learning",
        "body": """Neural networks are computing systems inspired by biological neural networks.
They consist of layers of interconnected nodes (neurons) that process
information using mathematical operations.

A typical neural network has:
- Input layer: Receives the raw data
- Hidden layers: Process the data through weighted connections
- Output layer: Produces the final prediction

Deep learning refers to neural networks with many hidden layers. Key
architectures include:

Convolutional Neural Networks (CNNs):
Designed for processing grid-like data such as images. They use
convolutional layers to automatically learn spatial hierarchies of features.
Applications: image classification, object detection, facial recognition.

Recurrent Neural Networks (RNNs):
Designed for sequential data. They maintain a hidden state that captures
information about previous elements in the sequence. Variants include
LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit).
Applications: natural language processing, time series prediction.

Transformers:
A newer architecture based on self-attention mechanisms. They process all
elements in a sequence simultaneously, making them highly parallelizable.
Applications: language models (GPT, BERT), machine translation, text
generation. Transformers have largely replaced RNNs for NLP tasks.""",
    },
    {
        "title": "Model Evaluation and Metrics",
        "body": """Evaluating machine learning models is crucial for understanding their
performance and ensuring they generalize well to unseen data.

Classification Metrics:
- Accuracy: Proportion of correct predictions (can be misleading for
  imbalanced datasets)
- Precision: Of all positive predictions, how many were actually positive
- Recall: Of all actual positives, how many were correctly identified
- F1 Score: Harmonic mean of precision and recall
- ROC-AUC: Area under the Receiver Operating Characteristic curve

Regression Metrics:
- MSE (Mean Squared Error): Average of squared differences
- RMSE (Root Mean Squared Error): Square root of MSE
- MAE (Mean Absolute Error): Average of absolute differences
- R-squared: Proportion of variance explained by the model

Cross-Validation:
K-fold cross-validation splits data into K subsets. The model is trained
on K-1 folds and tested on the remaining fold, rotating through all
combinations. This provides a more robust estimate of model performance.

Bias-Variance Tradeoff:
- High bias = underfitting (model too simple)
- High variance = overfitting (model too complex)
- Goal: Find the sweet spot that minimizes total error""",
    },
])

# ── Sample 2: Python Best Practices ──────────────────────────────────
create_pdf("python-best-practices.pdf", [
    {
        "title": "Python Code Style and PEP 8",
        "body": """PEP 8 is the official style guide for Python code. Following it makes your
code more readable and consistent with the broader Python ecosystem.

Key PEP 8 guidelines:

Indentation: Use 4 spaces per indentation level. Never mix tabs and spaces.

Line Length: Limit all lines to 79 characters for code, 72 for comments
and docstrings. Use parentheses for implicit line continuation.

Imports: Always put imports at the top of the file. Group them in this order:
1. Standard library imports
2. Related third-party imports
3. Local application/library specific imports

Naming Conventions:
- Variables and functions: snake_case
- Classes: CamelCase
- Constants: UPPER_SNAKE_CASE
- Private attributes: _leading_underscore
- Name-mangled attributes: __double_leading_underscore

Whitespace:
- No extra spaces inside parentheses, brackets, or braces
- One space around assignment operators
- No space before a colon in slices

Type Hints (PEP 484):
Python 3.5+ supports type hints. They make code more readable and enable
static analysis tools like mypy to catch type errors before runtime.

Example: def greet(name: str) -> str:
             return f"Hello, {name}" """,
    },
    {
        "title": "Error Handling and Testing",
        "body": """Proper error handling and testing are essential for robust Python code.

Exception Handling:
- Use specific exception types, not bare except clauses
- Use try/except/else/finally for comprehensive error handling
- Create custom exception classes for domain-specific errors
- Use context managers (with statement) for resource management
- Never silently swallow exceptions

Example pattern:
try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    raise
except ConnectionError:
    retry_with_backoff()
else:
    save_result(result)
finally:
    cleanup_resources()

Testing with pytest:
- Write tests in files named test_*.py
- Use descriptive test function names: test_user_creation_with_valid_email
- Use fixtures for shared setup code
- Use parametrize for testing multiple inputs
- Aim for high code coverage but focus on critical paths
- Use mocking to isolate units of code

Test structure (Arrange-Act-Assert):
def test_calculate_total():
    # Arrange
    items = [Item(price=10), Item(price=20)]
    # Act
    total = calculate_total(items)
    # Assert
    assert total == 30""",
    },
])

# ── Sample 3: RAG Architecture Guide ─────────────────────────────────
create_pdf("rag-architecture-guide.pdf", [
    {
        "title": "What is Retrieval-Augmented Generation (RAG)?",
        "body": """Retrieval-Augmented Generation (RAG) is a technique that enhances Large
Language Models (LLMs) by providing them with relevant external knowledge
retrieved from a document store at query time.

Why RAG?
- LLMs have a knowledge cutoff date and can hallucinate
- Fine-tuning is expensive and doesn't scale well for frequently updated data
- RAG provides grounded, up-to-date, and verifiable answers

The RAG Pipeline:
1. Document Ingestion: Load documents (PDF, HTML, text, etc.)
2. Text Extraction: Parse documents into plain text
3. Chunking: Split text into manageable pieces (typically 200-1000 chars)
4. Embedding: Convert chunks into dense vector representations
5. Indexing: Store vectors in a vector database with metadata
6. Query Processing: Convert user query into a vector
7. Retrieval: Find the most similar chunks using vector similarity search
8. Augmentation: Combine retrieved chunks with the user's query
9. Generation: LLM generates an answer grounded in the retrieved context

Key Design Decisions:
- Chunk size: Smaller chunks are more precise but may lack context
- Overlap: Overlapping chunks prevent information loss at boundaries
- Top-K: Number of chunks to retrieve (typically 3-10)
- Similarity metric: Cosine similarity is most common for normalized vectors""",
    },
    {
        "title": "Vector Databases and Embedding Models",
        "body": """Vector databases are specialized systems for storing and querying high-
dimensional vectors efficiently. They enable fast similarity search over
millions of vectors.

Popular Vector Databases:
- pgvector: PostgreSQL extension, great for existing Postgres users
- Pinecone: Fully managed cloud service
- Weaviate: Open-source with hybrid search capabilities
- Qdrant: Open-source with advanced filtering
- ChromaDB: Lightweight, good for prototyping
- Milvus: Open-source, designed for billion-scale vectors

pgvector Index Types:
- IVFFlat: Inverted file index, faster build time, good for smaller datasets
- HNSW: Hierarchical Navigable Small World, better recall, recommended for
  production use. Supports cosine, L2, and inner product distances.

Embedding Models:
The choice of embedding model significantly affects retrieval quality.

Key considerations:
- Dimension size: Higher dimensions capture more nuance (512-4096 typical)
- Training data: Models trained on similar domains perform better
- Asymmetric models: Use different encoders for queries vs passages
  (e.g., NVIDIA NV-EmbedQA uses input_type "query" vs "passage")
- Multilingual support: Important for non-English documents

Popular models: OpenAI text-embedding-3, Cohere Embed v3, NVIDIA NV-EmbedQA,
BGE, E5, and sentence-transformers models on HuggingFace.""",
    },
    {
        "title": "Advanced RAG Techniques",
        "body": """Beyond basic RAG, several techniques can improve retrieval quality and
answer accuracy.

Hybrid Search:
Combine dense vector search with traditional keyword search (BM25).
This captures both semantic similarity and exact keyword matches.

Re-ranking:
After initial retrieval, use a cross-encoder model to re-rank results.
Cross-encoders are more accurate than bi-encoders but slower, so they
are applied only to the top-K candidates.

Query Transformation:
- Query expansion: Add related terms to improve recall
- HyDE (Hypothetical Document Embeddings): Generate a hypothetical
  answer first, then use its embedding for retrieval
- Multi-query: Generate multiple query variations and merge results

Chunking Strategies:
- Fixed-size: Simple, consistent chunk sizes
- Sentence-based: Split on sentence boundaries
- Semantic: Use embedding similarity to find natural breakpoints
- Recursive: Try multiple separators (paragraphs, sentences, words)
- Parent-child: Store small chunks for retrieval but return larger
  parent chunks for context

Metadata Filtering:
Store metadata (source, date, author, topic) with chunks and use it
to filter results before or after vector search.

Evaluation:
Use frameworks like RAGAS to measure:
- Faithfulness: Is the answer grounded in the retrieved context?
- Relevancy: Are the retrieved chunks relevant to the query?
- Context precision: How precise is the retrieval?
- Answer correctness: Is the final answer correct?""",
    },
])

print("\nDone! Sample PDFs created in sample-pdfs/")
