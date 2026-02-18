"""
Seed script — 10 technical documents for the IBM RAG Guided project.
Run: python seed.py
"""

import asyncio
import sys
import os

# Add parent path for shared venv
sys.path.insert(0, os.path.dirname(__file__))

from app import init_db, get_conn, chunk_text, get_embeddings_batch

DOCUMENTS = [
    {
        "title": "Introduction to Transformer Architecture",
        "category": "machine learning",
        "content": """The Transformer architecture, introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. in 2017, revolutionized natural language processing and beyond. Unlike recurrent neural networks (RNNs) that process sequences step by step, Transformers process entire sequences in parallel using a mechanism called self-attention.

The core innovation is the multi-head attention mechanism, which allows the model to attend to different positions in the input sequence simultaneously. Each attention head learns different types of relationships between tokens. The attention function computes a weighted sum of values, where weights are determined by the compatibility between queries and keys.

The architecture consists of an encoder and decoder, each composed of stacked layers. Each encoder layer has two sub-layers: multi-head self-attention and a position-wise feed-forward network. The decoder adds a third sub-layer for cross-attention over the encoder output. Layer normalization and residual connections are used throughout.

Positional encoding is added to input embeddings since Transformers lack inherent sequence order awareness. The original paper used sinusoidal positional encodings, though learned positional embeddings are also common. Modern variants like RoPE (Rotary Position Embedding) offer improved extrapolation to longer sequences.

Transformers scale remarkably well with data and compute, leading to the development of large language models (LLMs) like GPT, BERT, T5, and their successors. The architecture has been adapted for computer vision (Vision Transformer), audio processing, protein structure prediction (AlphaFold), and many other domains. Pre-training on large corpora followed by fine-tuning on specific tasks has become the dominant paradigm in AI.""",
    },
    {
        "title": "Data Pipeline Architecture with Apache Kafka",
        "category": "data engineering",
        "content": """Apache Kafka is a distributed event streaming platform designed for high-throughput, fault-tolerant data pipelines. Originally developed at LinkedIn, Kafka has become the backbone of real-time data infrastructure at thousands of organizations.

Kafka's architecture centers on the concept of topics, which are partitioned logs. Producers write messages to topic partitions, and consumers read from them. Each partition is an ordered, immutable sequence of messages, each assigned a sequential offset. This design enables both high throughput and message ordering guarantees within partitions.

Key architectural components include brokers (servers that store data and serve clients), ZooKeeper or KRaft for cluster coordination, and the replication protocol for fault tolerance. Each partition has one leader broker and multiple follower replicas. Producers can choose between acknowledgment levels: acks=0 (fire and forget), acks=1 (leader acknowledgment), and acks=all (full ISR acknowledgment).

Consumer groups enable parallel processing. Each partition is assigned to exactly one consumer in a group, allowing horizontal scaling. The consumer offset tracking mechanism allows consumers to resume from where they left off after failures.

Kafka Connect provides a framework for connecting Kafka with external systems through source and sink connectors. Popular connectors exist for databases (Debezium for CDC), cloud storage (S3, GCS), search engines (Elasticsearch), and data warehouses (Snowflake, BigQuery).

Kafka Streams and ksqlDB offer stream processing capabilities directly on Kafka. These tools enable real-time transformations, aggregations, joins, and windowed computations without requiring a separate processing cluster. The exactly-once semantics (EOS) feature ensures data consistency in stream processing applications.

Schema Registry manages Avro, Protobuf, or JSON Schema schemas for Kafka topics, enabling schema evolution while maintaining compatibility between producers and consumers.""",
    },
    {
        "title": "Kubernetes Production Best Practices",
        "category": "cloud architecture",
        "content": """Kubernetes (K8s) is the de facto standard for container orchestration in production environments. Running workloads reliably at scale requires careful attention to resource management, security, observability, and deployment strategies.

Resource management starts with setting appropriate requests and limits for CPU and memory. Requests determine scheduling decisions and guarantee minimum resources, while limits prevent containers from consuming excessive resources. The Vertical Pod Autoscaler (VPA) can recommend optimal resource values based on actual usage. Horizontal Pod Autoscaler (HPA) scales the number of replicas based on metrics like CPU utilization, memory usage, or custom metrics from Prometheus.

Pod Disruption Budgets (PDBs) ensure application availability during voluntary disruptions like node upgrades. They specify the minimum number or percentage of pods that must remain available. Combined with pod anti-affinity rules, PDBs help distribute workloads across failure domains.

Security best practices include running containers as non-root users, using read-only root filesystems, implementing network policies to restrict pod-to-pod communication, and using Pod Security Standards (previously Pod Security Policies). RBAC should follow the principle of least privilege. Secrets should be managed through external secret stores like HashiCorp Vault or AWS Secrets Manager rather than stored directly in etcd.

Observability requires a three-pillar approach: metrics (Prometheus + Grafana), logs (Fluentd/Fluent Bit + Elasticsearch/Loki), and traces (Jaeger/Tempo with OpenTelemetry). The combination provides comprehensive visibility into application behavior and performance.

Deployment strategies include rolling updates (default), blue-green deployments (using service switching), and canary releases (using service mesh or ingress controllers). GitOps tools like ArgoCD or Flux automate deployments by synchronizing cluster state with Git repositories. Progressive delivery tools like Flagger automate canary analysis and promotion.""",
    },
    {
        "title": "Vector Databases and Similarity Search",
        "category": "machine learning",
        "content": """Vector databases are purpose-built systems for storing, indexing, and querying high-dimensional vector embeddings. They are essential components in modern AI applications, particularly in retrieval-augmented generation (RAG), recommendation systems, and similarity search.

The fundamental operation is approximate nearest neighbor (ANN) search, which finds vectors similar to a query vector efficiently. Exact nearest neighbor search has O(n) complexity, making it impractical for large datasets. ANN algorithms trade small accuracy losses for dramatic speed improvements.

Popular indexing algorithms include HNSW (Hierarchical Navigable Small World), which builds a multi-layer graph structure for efficient traversal. HNSW offers excellent query performance with high recall, making it the most popular choice for production workloads. IVF (Inverted File Index) partitions the vector space into clusters and searches only relevant clusters. Product Quantization (PQ) compresses vectors to reduce memory usage at the cost of some accuracy.

pgvector extends PostgreSQL with vector similarity search capabilities. It supports exact and approximate nearest neighbor search using IVF and HNSW indexes. The advantage of pgvector is co-locating vector search with relational data, eliminating the need for a separate vector database. It supports L2 distance, inner product, and cosine distance operators.

Dedicated vector databases like Pinecone, Weaviate, Qdrant, Milvus, and Chroma offer additional features like metadata filtering, multi-tenancy, horizontal scaling, and hybrid search (combining vector and keyword search). Choosing between pgvector and a dedicated solution depends on scale requirements, operational complexity tolerance, and feature needs.

Embedding quality significantly impacts search results. Models like OpenAI text-embedding-3, Cohere embed-v3, and open-source models like E5, BGE, and GTE produce embeddings optimized for retrieval. The choice of embedding model, chunk size, and overlap strategy all affect retrieval quality in RAG systems.""",
    },
    {
        "title": "MLOps: Machine Learning Operations in Practice",
        "category": "machine learning",
        "content": """MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently. It addresses the unique challenges of ML systems, including data dependencies, model decay, and the experimental nature of ML development.

The ML lifecycle includes data collection, data validation, feature engineering, model training, model evaluation, model deployment, and monitoring. Each stage presents unique challenges. Data validation ensures training data quality and detects data drift. Feature stores like Feast or Tecton provide consistent feature computation across training and serving.

Model versioning and experiment tracking are fundamental. Tools like MLflow, Weights & Biases, and Neptune track experiments including hyperparameters, metrics, artifacts, and code versions. DVC (Data Version Control) extends Git to handle large datasets and model files. Model registries provide a central catalog of trained models with metadata and lifecycle stage tracking.

CI/CD for ML extends traditional CI/CD with additional pipelines for data validation, model training, model evaluation, and model deployment. Tools like Kubeflow Pipelines, Apache Airflow, and Prefect orchestrate complex ML workflows. GitHub Actions and GitLab CI integrate ML pipeline triggers with code changes.

Model serving options range from simple REST APIs (FastAPI, Flask) to dedicated serving platforms like TensorFlow Serving, Triton Inference Server, and Seldon Core. Key considerations include latency requirements, throughput, model format support, and GPU utilization. Batching strategies, model caching, and quantization optimize serving performance.

Monitoring in production includes tracking prediction quality (accuracy degradation), data drift (input distribution changes), concept drift (relationship changes between inputs and outputs), and system metrics (latency, throughput, errors). Automated retraining pipelines can trigger when drift is detected, maintaining model quality over time.""",
    },
    {
        "title": "Retrieval-Augmented Generation (RAG) Architecture",
        "category": "machine learning",
        "content": """Retrieval-Augmented Generation (RAG) is an architecture pattern that enhances large language models (LLMs) by providing them with relevant external knowledge at inference time. Rather than relying solely on knowledge encoded in model parameters during pre-training, RAG systems retrieve relevant documents and include them in the LLM's context.

The basic RAG pipeline consists of three stages: indexing, retrieval, and generation. During indexing, documents are split into chunks, embedded using an embedding model, and stored in a vector database. At query time, the user's question is embedded and used to find similar chunks via approximate nearest neighbor search. The retrieved chunks are then included in the LLM prompt as context for answer generation.

Advanced RAG patterns address limitations of the basic approach. Query expansion generates multiple variations of the user's question to improve recall. Hypothetical Document Embeddings (HyDE) generates a hypothetical answer and uses its embedding for retrieval. Re-ranking uses a cross-encoder or LLM to re-score retrieved chunks by true relevance rather than embedding similarity alone.

Chunking strategy significantly impacts retrieval quality. Fixed-size chunking is simple but may split semantic units. Semantic chunking uses sentence boundaries and topic changes to create more meaningful chunks. Hierarchical chunking creates parent-child chunk relationships, allowing retrieval of specific passages while maintaining broader context. Typical chunk sizes range from 256 to 1024 tokens with 10-20% overlap.

Evaluation of RAG systems requires multiple metrics. Retrieval quality is measured by precision, recall, and MRR (Mean Reciprocal Rank). Generation quality includes faithfulness (is the answer grounded in retrieved context?), relevance (does the answer address the question?), and harmfulness (does the answer contain problematic content?). Frameworks like RAGAS, TruLens, and DeepEval provide automated evaluation pipelines.

Production considerations include caching frequently asked queries, implementing fallback strategies when retrieval quality is low, handling multi-turn conversations with context management, and monitoring retrieval and generation quality in real-time. IBM's production RAG approach emphasizes systematic evaluation, query understanding, and answer verification as critical components for enterprise deployment.""",
    },
    {
        "title": "PostgreSQL Performance Tuning Guide",
        "category": "data engineering",
        "content": """PostgreSQL is a powerful open-source relational database that can handle diverse workloads from OLTP to analytics. Proper tuning is essential for optimal performance at scale.

Memory configuration is the first optimization area. shared_buffers should typically be set to 25% of system RAM (e.g., 4GB for a 16GB system). effective_cache_size should reflect the total memory available for caching (typically 50-75% of RAM). work_mem controls memory for sort operations and hash tables; setting it too high with many concurrent connections can cause memory exhaustion. maintenance_work_mem affects operations like VACUUM, CREATE INDEX, and ALTER TABLE.

Query optimization starts with understanding EXPLAIN ANALYZE output. Key metrics include actual time, rows, and loops for each plan node. Sequential scans on large tables often indicate missing indexes. The planner's cost estimates can be improved by ensuring statistics are up to date (regular ANALYZE runs or autovacuum statistics collection).

Indexing strategies include B-tree (default, for equality and range queries), GIN (for full-text search, arrays, JSONB), GiST (for geometric and range types), and HNSW/IVF (for vector similarity via pgvector). Partial indexes reduce index size by only indexing rows matching a predicate. Expression indexes index computed values. Covering indexes (INCLUDE clause) enable index-only scans.

Connection pooling with PgBouncer or Pgpool-II is essential for applications with many concurrent connections. PostgreSQL creates a process per connection, making connection overhead significant. PgBouncer in transaction mode allows hundreds of application connections to share a smaller pool of database connections.

Partitioning improves query performance and maintenance for large tables. Range partitioning (by date) is most common. The planner can eliminate irrelevant partitions during query planning (partition pruning). Partitioning also enables parallel maintenance operations like VACUUM on individual partitions.

VACUUM and autovacuum management prevents table bloat and maintains query performance. Dead tuples from updates and deletes accumulate until vacuumed. Autovacuum should be tuned for the workload: aggressive settings for write-heavy tables, relaxed for read-mostly tables. VACUUM FREEZE prevents transaction ID wraparound, which can cause database shutdown if not addressed.""",
    },
    {
        "title": "Microservices Communication Patterns",
        "category": "cloud architecture",
        "content": """Microservices architecture decomposes applications into small, independently deployable services. Inter-service communication is a critical design decision that impacts reliability, performance, and complexity.

Synchronous communication patterns include REST (HTTP/JSON), gRPC (Protocol Buffers over HTTP/2), and GraphQL. REST is the most common, offering simplicity and broad tooling support. gRPC provides better performance through binary serialization and HTTP/2 multiplexing, making it preferred for internal service-to-service calls. GraphQL excels at aggregating data from multiple services into a single API for frontend consumption.

Asynchronous messaging patterns decouple services temporally. Message brokers like Apache Kafka, RabbitMQ, and AWS SQS/SNS enable event-driven architectures. Event-driven communication allows services to react to state changes without direct coupling. The Saga pattern coordinates distributed transactions across services using choreography (events) or orchestration (a coordinator service).

Service mesh technologies like Istio, Linkerd, and Consul Connect provide infrastructure-level communication features including mutual TLS, load balancing, circuit breaking, and observability without changing application code. The sidecar proxy pattern intercepts all network traffic, enabling policy enforcement and telemetry collection.

The Circuit Breaker pattern prevents cascade failures by monitoring failure rates and temporarily stopping requests to failing services. When a service exceeds a failure threshold, the circuit opens and returns immediate errors rather than waiting for timeouts. After a reset interval, the circuit half-opens and allows test requests through.

API Gateway pattern provides a single entry point for external clients. Gateways handle cross-cutting concerns like authentication, rate limiting, request routing, and response aggregation. Tools like Kong, Ambassador, and AWS API Gateway implement this pattern. The Backend for Frontend (BFF) pattern creates specialized gateways for different client types (web, mobile).

Service discovery enables services to locate each other dynamically. DNS-based discovery (Kubernetes Services) is simplest. Client-side discovery (Netflix Eureka) gives clients direct access to service instances. Server-side discovery (load balancers) simplifies client logic but adds a network hop. Health checking ensures traffic only routes to healthy instances.""",
    },
    {
        "title": "Feature Engineering for Machine Learning",
        "category": "machine learning",
        "content": """Feature engineering is the process of transforming raw data into features that better represent the underlying patterns for machine learning models. Despite advances in deep learning that automate some feature extraction, thoughtful feature engineering remains crucial for many practical ML applications.

Numerical feature transformations include standardization (zero mean, unit variance), min-max normalization (scaling to [0,1]), log transformation (for skewed distributions), and power transformations (Box-Cox, Yeo-Johnson). Binning converts continuous variables into categorical ones, which can capture non-linear relationships in linear models. Polynomial features create interaction terms and higher-order features.

Categorical encoding strategies go beyond one-hot encoding. Target encoding replaces categories with the mean of the target variable, reducing dimensionality for high-cardinality features. Frequency encoding uses category occurrence counts. Ordinal encoding preserves natural ordering when it exists. Hash encoding (feature hashing) handles very high cardinality by mapping categories to a fixed-size vector.

Temporal features extract patterns from timestamps: hour of day, day of week, month, quarter, and cyclical encodings (sine/cosine transformations for periodic features). Lag features capture previous values in time series. Rolling statistics (mean, standard deviation, min, max over sliding windows) capture trends and volatility. Date differences and durations provide temporal context.

Text feature engineering includes TF-IDF vectorization, word embeddings (Word2Vec, GloVe, FastText), and sentence embeddings from transformer models. Text length, word count, punctuation density, and readability scores serve as supplementary features. Named entity extraction, sentiment scores, and topic model outputs provide semantic features.

Feature selection methods reduce dimensionality and remove noisy features. Filter methods (mutual information, chi-squared, correlation analysis) evaluate features independently. Wrapper methods (recursive feature elimination, forward/backward selection) evaluate feature subsets. Embedded methods (L1 regularization, tree-based feature importance) perform selection during model training. SHAP values provide model-agnostic feature importance for any trained model.

Feature stores provide a centralized platform for feature management. They ensure consistency between training and serving features (training-serving skew prevention), enable feature sharing across teams and models, and maintain feature lineage and documentation. Point-in-time correctness prevents data leakage by ensuring features are computed using only data available at prediction time.""",
    },
    {
        "title": "Observability in Distributed Systems",
        "category": "cloud architecture",
        "content": """Observability is the ability to understand a system's internal state by examining its outputs. In distributed systems, observability is built on three pillars: metrics, logs, and traces. Together, they provide the context needed to debug issues, understand performance, and ensure reliability.

Metrics are numerical measurements collected over time. They are efficient to store and query, making them ideal for dashboards and alerting. Key metric types include counters (monotonically increasing values like request counts), gauges (point-in-time values like queue depth), and histograms (distribution of values like response times). The RED method (Rate, Errors, Duration) provides a standard set of metrics for request-driven services. The USE method (Utilization, Saturation, Errors) focuses on infrastructure resources.

Prometheus is the de facto standard for metrics collection in cloud-native environments. It uses a pull-based model, scraping metrics endpoints at regular intervals. PromQL enables powerful queries and aggregations. Grafana provides visualization with customizable dashboards. AlertManager handles alert routing, grouping, and silencing. For long-term storage, solutions like Thanos, Cortex, or Mimir extend Prometheus with horizontal scaling and multi-tenancy.

Structured logging replaces unstructured text logs with machine-parseable formats (typically JSON). Each log entry includes contextual fields like request ID, user ID, service name, and trace ID. Log aggregation pipelines typically use Fluent Bit or Fluentd for collection, with Elasticsearch/OpenSearch or Grafana Loki for storage and search. Log levels (DEBUG, INFO, WARN, ERROR) control verbosity and aid filtering.

Distributed tracing tracks requests as they propagate across service boundaries. Each trace consists of spans representing individual operations. OpenTelemetry provides a vendor-neutral SDK for instrumenting applications. Trace context propagation (W3C Trace Context standard) ensures correlation across services. Backends like Jaeger, Tempo, and Zipkin store and visualize traces. Sampling strategies (head-based, tail-based) reduce storage costs while preserving interesting traces.

Service Level Objectives (SLOs) formalize reliability targets. An SLI (Service Level Indicator) is a metric measuring service behavior (e.g., 99th percentile latency). An SLO defines a target for that metric (e.g., p99 latency < 200ms). Error budgets quantify the acceptable amount of unreliability, enabling teams to balance feature velocity with reliability. SLO-based alerting (burn rate alerts) reduces alert fatigue by focusing on threats to the error budget rather than individual incidents.

Correlation across pillars is critical. Exemplars link metrics to specific traces. Trace IDs in logs connect log entries to their distributed trace. This cross-referencing enables rapid root cause analysis: alert on a metric anomaly, drill into traces showing slow requests, and examine logs from the specific service and time window.""",
    },
]


async def seed():
    """Seed the database with sample documents."""
    print("Initializing database...")
    init_db()

    # Check if already seeded
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ibm_documents")
            count = cur.fetchone()[0]
            if count > 0:
                print(f"Database already has {count} documents. Skipping seed.")
                print("To re-seed, run: DELETE FROM ibm_documents;")
                return

    print(f"Seeding {len(DOCUMENTS)} documents...")

    for i, doc in enumerate(DOCUMENTS):
        print(f"  [{i+1}/{len(DOCUMENTS)}] {doc['title']}...")

        # Chunk the document
        chunks = chunk_text(doc["content"])
        print(f"    → {len(chunks)} chunks")

        # Get embeddings
        embeddings = await get_embeddings_batch([c for c in chunks], input_type="passage")
        print(f"    → {len(embeddings)} embeddings")

        # Store in DB
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ibm_documents (title, content, category) VALUES (%s, %s, %s) RETURNING id",
                    (doc["title"], doc["content"], doc["category"]),
                )
                doc_id = cur.fetchone()[0]

                for chunk_content, emb in zip(chunks, embeddings):
                    emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                    cur.execute(
                        "INSERT INTO ibm_chunks (document_id, content, embedding) VALUES (%s, %s, %s::vector)",
                        (doc_id, chunk_content, emb_str),
                    )
            conn.commit()

        print(f"    ✓ Stored (doc_id={doc_id})")

    print("\n✅ Seeding complete!")

    # Show summary
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ibm_documents")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ibm_chunks")
            chunk_count = cur.fetchone()[0]
    print(f"   Documents: {doc_count}")
    print(f"   Chunks: {chunk_count}")


if __name__ == "__main__":
    asyncio.run(seed())
