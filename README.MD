# Enterprise Knowledge Graph

A knowledge graph system demonstrating entity resolution, relationship inference, and RAG (Retrieval-Augmented Generation) with intentional contradictions for validation.

## Features

- **Multi-source data ingestion**: CSV, DOCX, TXT, PDF
- **Entity resolution**: Canonical entities vs. external entities (GDPR, AWS, Salesforce terms)
- **Neo4j graph database**: 598 nodes, 777 relationships
- **Semantic search**: FAISS vector indexes
- **RAG pipeline**: Ollama-powered Q&A with source attribution
- **Streamlit frontend**: Professional chat interface
- **Comprehensive validation**: Golden query testing with quality metrics

## Architecture

```
Data Generation (Phases 1-5)
    ↓
Neo4j Ingestion (Phase 6)
    ↓
Entity Extraction (Phase 7: spaCy NER + resolution)
    ↓
Embeddings (Phase 8: SentenceTransformers + FAISS)
    ↓
RAG Pipeline (Phase 9: Retrieval + Ollama generation)
    ↓
Validation (Phase 10: Golden queries)
    ↓
Frontend (Phase 11: Streamlit)
```

## Prerequisites

- **Python 3.9+**
- **Neo4j Desktop** (or Docker)
- **Ollama** with `llama3.2:3b` model
- **16GB RAM recommended**
- **GPU optional** (for faster embeddings)

## Quick Start

### 1. Install Dependencies

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install Python packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_md
```

### 2. Setup Services

**Neo4j:**
```bash
# Option A: Neo4j Desktop
# - Download from https://neo4j.com/download/
# - Create database, start, note password

# Option B: Docker
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

**Ollama:**
```bash
# Install Ollama (https://ollama.ai)
# macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2:3b
```

### 3. Configure

Edit `config/neo4j.yaml`:
```yaml
uri: "bolt://localhost:7687"
user: "neo4j"
password: "YOUR_PASSWORD"  # ← Change this
```

### 4. Run Pipeline

```bash
# Generate synthetic data (Phases 1-5)
python run_data_gen.py

# Build knowledge graph (Phases 6-9: ~5-10 minutes)
python run_pipeline.py

# Validate system (Phase 10: optional)
python validate_rag.py

# Launch frontend (Phase 11)
streamlit run streamlit_app.py
```

Access UI at: **http://localhost:8501**

## Usage

**Example queries:**
- "Who is Debra's manager?"
- "What department is Anthony in?"
- "What projects use AWS?"
- "Tell me about the Jira Cloud Adoption project"
- "What does GDPR say about data retention?"

## Project Structure

```
├── config/                   # Configuration files
│   ├── pipeline.yaml        # Master config
│   ├── neo4j.yaml          # Database credentials
│   └── ollama.yaml         # LLM settings
├── data/                    # Generated data & outputs
│   ├── source/             # Raw data (CSV, DOCX, TXT, PDF)
│   ├── processed/          # Entity registry, metadata
│   ├── graph/              # Neo4j exports
│   ├── knowledge/          # Extracted triples
│   └── embeddings/         # FAISS indexes
├── gen_data_*.py           # Data generators (Phases 1-5)
├── neo4j_loader.py         # Phase 6: Graph ingestion
├── entity_extractor.py     # Phase 7: NER + resolution
├── embedding_generator.py  # Phase 8: Embeddings
├── rag_system.py           # Phase 9: RAG pipeline
├── validate_rag.py         # Phase 10: Validation
├── streamlit_app.py        # Phase 11: Frontend
├── run_data_gen.py         # Orchestrator (Phases 1-5)
├── run_pipeline.py         # Orchestrator (Phases 6-9)
└── requirements.txt        # Python dependencies
```

## Configuration

Key settings in `config/pipeline.yaml`:

```yaml
extraction:
  resolution_threshold: 0.90  # Entity matching strictness
  cooccurrence_threshold: 3   # Min co-occurrences for relationships

embedding:
  model_name: "all-MiniLM-L6-v2"
  chunk_size: 500

rag:
  retrieval:
    triple_top_k: 10
    chunk_top_k: 15
    similarity_threshold: 0.15
```

## Key Design Choices

1. **Sparse graph**: ~1.3 edges/node (realistic enterprise data)
2. **External entity isolation**: 515 external entities (GDPR, AWS, Salesforce) isolated from canonical graph
3. **Structured triple format**: Enables precise FAISS retrieval
4. **Anti-hallucination**: Returns "I don't know" when sources insufficient
5. **Contradiction surfacing**: Intentional contradictions flagged for validation

## Validation

```bash
# Run golden query validation
python validate_rag.py --detailed

# View results
cat data/logs/validation/validation_report.json
```

**Success criteria:**
- Overall accuracy: ≥75%
- Graph queries: ≥80%
- Semantic queries: ≥60%
- Contradiction detection: 100%

## Troubleshooting

**"Neo4j connection failed"**
- Ensure Neo4j is running: `neo4j status`
- Check credentials in `config/neo4j.yaml`

**"Ollama model not found"**
```bash
ollama list  # Check installed models
ollama pull llama3.2:3b  # Install if missing
```

**"FAISS index not found"**
```bash
# Rebuild embeddings (Phase 8)
python embedding_generator.py
```

**"Low validation accuracy"**
- Adjust retrieval thresholds in `config/pipeline.yaml`
- Increase `triple_top_k` and `chunk_top_k`
- Lower `similarity_threshold`

## Tech Stack

- **Graph DB**: Neo4j 5.x
- **NER**: spaCy (`en_core_web_md`)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS (IndexFlatIP)
- **LLM**: Ollama (`llama3.2:3b`)
- **Frontend**: Streamlit
- **Data Processing**: pandas, python-docx, PyMuPDF

## Performance

- **Data generation**: ~2 minutes
- **Neo4j ingestion**: ~30 seconds
- **Entity extraction**: ~3 minutes
- **Embedding generation**: ~2 minutes (CPU)
- **Query latency**: 500-2000ms (P95)

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- Based on enterprise knowledge graph patterns
- spaCy for NER
- Anthropic Claude for validation design
- Neo4j graph database platform

## Citation

If you use this project, please cite:

```bibtex
@misc{codeflow_kg_2025,
  title={CodeFlow Corp Knowledge Graph Demo},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/YOUR_REPO_NAME}
}
```