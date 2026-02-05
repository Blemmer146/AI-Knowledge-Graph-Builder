"""
Phase 8: Embeddings & FAISS Indexing
Creates semantic search capability for RAG pipeline

Author: Enterprise KG Project
Date: 2025-01-25
"""

import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import sys
from collections import defaultdict

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed. Run: pip install neo4j")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu not installed. Run: pip install faiss-cpu")
    sys.exit(1)


class EmbeddingGenerator:
    """Generate embeddings and FAISS indexes for triples and document chunks"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize embedding generation pipeline

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model_name: SentenceTransformer model name
            chunk_size: Words per chunk
            chunk_overlap: Overlapping words between chunks
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.driver = None
        self.model = None
        self.config = None

        # Data storage
        self.triples = []
        self.triple_texts = []
        self.triple_metadata = []

        self.documents = []
        self.chunks = []
        self.chunk_texts = []
        self.chunk_metadata = []

        # Embeddings
        self.triple_embeddings = None
        self.chunk_embeddings = None

        # FAISS indexes
        self.triple_index = None
        self.chunk_index = None

        # Statistics
        self.stats = defaultdict(int)

        # Output paths
        self.data_dir = Path("data")
        self.knowledge_dir = self.data_dir / "knowledge"  # CHANGED
        self.output_dir = self.data_dir / "embeddings"  # CHANGED
        self.output_dir.mkdir(exist_ok=True)

        print(f"Embedding Generation Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Chunk size: {chunk_size} words")
        print(f"  Chunk overlap: {chunk_overlap} words")



    def connect_neo4j(self):
        """Connect to Neo4j"""
        print(f"\nConnecting to Neo4j at {self.neo4j_uri}...")
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            with self.driver.session() as s:
                s.run("RETURN 1").single()
            print("Neo4j connection successful")
        except Exception as e:
            print(f"ERROR: Cannot connect to Neo4j - {e}")
            sys.exit(1)

    def load_model(self):
        """Load SentenceTransformer model"""
        print(f"\nLoading SentenceTransformer model '{self.model_name}'...")
        try:
            self.model = SentenceTransformer(self.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded (embedding dimension: {embedding_dim})")
            self.stats['embedding_dimension'] = embedding_dim
        except Exception as e:
            print(f"ERROR: Cannot load model - {e}")
            sys.exit(1)

    def load_triples(self):
        """Load triples from Phase 7"""
        print("\n=== STEP 1: Loading Triples ===")

        triples_path = self.knowledge_dir / "triples.json"  # CHANGED PATH
        if not triples_path.exists():
            print(f"ERROR: {triples_path} not found. Run Phase 7 first.")
            sys.exit(1)

        with open(triples_path) as f:
            self.triples = json.load(f)

        print(f"Loaded {len(self.triples)} triples from Phase 7")

        # Convert to structured text format
        for i, triple in enumerate(self.triples):
            # Structured format for precise retrieval
            structured_text = (
                f"Subject: {triple['subject']['name']} | "
                f"Predicate: {triple['predicate']} | "
                f"Object: {triple['object']['name']} | "
                f"Confidence: {triple['confidence']}"
            )

            self.triple_texts.append(structured_text)

            # Metadata for retrieval
            metadata = {
                'id': f"triple_{i:04d}",
                'subject_id': triple['subject']['id'],
                'subject_name': triple['subject']['name'],
                'subject_type': triple['subject']['type'],
                'predicate': triple['predicate'],
                'object_id': triple['object']['id'],
                'object_name': triple['object']['name'],
                'object_type': triple['object']['type'],
                'confidence': triple['confidence'],
                'source': triple['source'],
                'flagged': triple.get('flagged', False),
                'inferred': triple.get('inferred', False),
                'structured_text': structured_text,
                'natural_text': triple['text']  # Keep for Phase 9 display
            }
            self.triple_metadata.append(metadata)

        self.stats['total_triples'] = len(self.triples)
        self.stats['ground_truth_triples'] = len([t for t in self.triples if t['confidence'] == 1.0])
        self.stats['inferred_triples'] = len([t for t in self.triples if t.get('inferred')])
        self.stats['flagged_triples'] = len([t for t in self.triples if t.get('flagged')])

        print(f"  Ground truth: {self.stats['ground_truth_triples']}")
        print(f"  Inferred: {self.stats['inferred_triples']}")
        print(f"  Flagged: {self.stats['flagged_triples']}")

    def load_documents(self):
        """Load documents from Neo4j"""
        print("\n=== STEP 2: Loading Documents ===")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                WHERE d.word_count > 0
                RETURN d.id AS doc_id,
                       d.filename AS filename,
                       d.full_text AS text,
                       d.type AS doc_type,
                       d.word_count AS word_count,
                       d.confidence AS confidence,
                       coalesce(d.has_contradictions, false) AS has_contradictions
                ORDER BY d.filename
            """)
            self.documents = [dict(record) for record in result]

        print(f"Loaded {len(self.documents)} documents")

        total_words = sum(d['word_count'] for d in self.documents)
        print(f"  Total words: {total_words:,}")

        self.stats['total_documents'] = len(self.documents)
        self.stats['total_words'] = total_words

    def chunk_documents(self):
        """Chunk documents with overlap"""
        print("\n=== STEP 3: Chunking Documents ===")

        for doc in self.documents:
            doc_id = doc['doc_id']
            text = doc['text']
            words = text.split()

            if len(words) <= self.chunk_size:
                # Document is smaller than chunk size - single chunk
                chunk = {
                    'id': f"{doc_id}_chunk_0000",
                    'doc_id': doc_id,
                    'doc_filename': doc['filename'],
                    'doc_type': doc['doc_type'],
                    'doc_confidence': doc['confidence'],
                    'has_contradictions': doc['has_contradictions'],
                    'text': text,
                    'word_count': len(words),
                    'chunk_index': 0,
                    'start_word': 0,
                    'end_word': len(words)
                }
                self.chunks.append(chunk)
                self.chunk_texts.append(text)
                self.chunk_metadata.append(chunk)
            else:
                # Split into overlapping chunks
                chunk_index = 0
                start = 0

                while start < len(words):
                    end = min(start + self.chunk_size, len(words))
                    chunk_words = words[start:end]
                    chunk_text = ' '.join(chunk_words)

                    chunk = {
                        'id': f"{doc_id}_chunk_{chunk_index:04d}",
                        'doc_id': doc_id,
                        'doc_filename': doc['filename'],
                        'doc_type': doc['doc_type'],
                        'doc_confidence': doc['confidence'],
                        'has_contradictions': doc['has_contradictions'],
                        'text': chunk_text,
                        'word_count': len(chunk_words),
                        'chunk_index': chunk_index,
                        'start_word': start,
                        'end_word': end
                    }
                    self.chunks.append(chunk)
                    self.chunk_texts.append(chunk_text)
                    self.chunk_metadata.append(chunk)

                    # Move forward with overlap
                    if end == len(words):
                        break
                    start += (self.chunk_size - self.chunk_overlap)
                    chunk_index += 1

        print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        print(f"  Average chunks per document: {len(self.chunks) / len(self.documents):.1f}")

        self.stats['total_chunks'] = len(self.chunks)

        # Breakdown by document type
        chunks_by_type = defaultdict(int)
        for chunk in self.chunks:
            chunks_by_type[chunk['doc_type']] += 1

        print("  Chunks by type:")
        for doc_type, count in sorted(chunks_by_type.items()):
            print(f"    {doc_type}: {count}")

    def generate_triple_embeddings(self):
        """Generate embeddings for triples"""
        print("\n=== STEP 4: Generating Triple Embeddings ===")

        print(f"Encoding {len(self.triple_texts)} triples...")
        self.triple_embeddings = self.model.encode(
            self.triple_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Generated embeddings: shape {self.triple_embeddings.shape}")
        self.stats['triple_embeddings_shape'] = list(self.triple_embeddings.shape)

    def generate_chunk_embeddings(self):
        """Generate embeddings for document chunks"""
        print("\n=== STEP 5: Generating Chunk Embeddings ===")

        print(f"Encoding {len(self.chunk_texts)} chunks...")
        self.chunk_embeddings = self.model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Generated embeddings: shape {self.chunk_embeddings.shape}")
        self.stats['chunk_embeddings_shape'] = list(self.chunk_embeddings.shape)

    def build_faiss_indexes(self):
        """Build FAISS indexes for exact search"""
        print("\n=== STEP 6: Building FAISS Indexes ===")

        # Normalize embeddings for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(self.triple_embeddings)
        faiss.normalize_L2(self.chunk_embeddings)

        # Build triple index
        embedding_dim = self.triple_embeddings.shape[1]
        self.triple_index = faiss.IndexFlatIP(embedding_dim)
        self.triple_index.add(self.triple_embeddings)
        print(f"Triple index built: {self.triple_index.ntotal} vectors")

        # Build chunk index
        self.chunk_index = faiss.IndexFlatIP(embedding_dim)
        self.chunk_index.add(self.chunk_embeddings)
        print(f"Chunk index built: {self.chunk_index.ntotal} vectors")

        self.stats['faiss_index_type'] = 'IndexFlatIP'
        self.stats['triple_index_size'] = self.triple_index.ntotal
        self.stats['chunk_index_size'] = self.chunk_index.ntotal

    def save_outputs(self):
        """Save embeddings, metadata, and indexes"""
        print("\n=== STEP 7: Saving Outputs ===")

        # Save triple embeddings
        triple_emb_path = self.output_dir / "triple_embeddings.npy"
        np.save(triple_emb_path, self.triple_embeddings)
        print(f"Saved triple embeddings: {triple_emb_path}")

        # Save triple metadata
        triple_meta_path = self.output_dir / "triple_metadata.json"
        with open(triple_meta_path, 'w') as f:
            json.dump(self.triple_metadata, f, indent=2)
        print(f"Saved triple metadata: {triple_meta_path}")

        # Save chunk embeddings
        chunk_emb_path = self.output_dir / "chunk_embeddings.npy"
        np.save(chunk_emb_path, self.chunk_embeddings)
        print(f"Saved chunk embeddings: {chunk_emb_path}")

        # Save chunk metadata
        chunk_meta_path = self.output_dir / "chunk_metadata.json"
        with open(chunk_meta_path, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)
        print(f"Saved chunk metadata: {chunk_meta_path}")

        # Save FAISS indexes
        triple_index_path = self.output_dir / "faiss_triples.index"
        faiss.write_index(self.triple_index, str(triple_index_path))
        print(f"Saved triple FAISS index: {triple_index_path}")

        chunk_index_path = self.output_dir / "faiss_chunks.index"
        faiss.write_index(self.chunk_index, str(chunk_index_path))
        print(f"Saved chunk FAISS index: {chunk_index_path}")

        # Save embedding log
        embedding_log = {
            'generation_timestamp': datetime.now().isoformat(),
            'configuration': {
                'model_name': self.model_name,
                'embedding_dimension': self.stats['embedding_dimension'],
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'faiss_index_type': 'IndexFlatIP'
            },
            'statistics': dict(self.stats),
            'triple_summary': {
                'total': self.stats['total_triples'],
                'ground_truth': self.stats['ground_truth_triples'],
                'inferred': self.stats['inferred_triples'],
                'flagged': self.stats['flagged_triples']
            },
            'chunk_summary': {
                'total_chunks': self.stats['total_chunks'],
                'total_documents': self.stats['total_documents'],
                'avg_chunks_per_doc': round(self.stats['total_chunks'] / self.stats['total_documents'], 2)
            },
            'outputs': {
                'triple_embeddings': str(triple_emb_path),
                'triple_metadata': str(triple_meta_path),
                'chunk_embeddings': str(chunk_emb_path),
                'chunk_metadata': str(chunk_meta_path),
                'triple_index': str(triple_index_path),
                'chunk_index': str(chunk_index_path)
            }
        }

        log_path = self.output_dir / "embedding_log.json"
        with open(log_path, 'w') as f:
            json.dump(embedding_log, f, indent=2)
        print(f"Saved embedding log: {log_path}")

    def validate_retrieval(self):
        """Test retrieval with sample queries"""
        print("\n=== STEP 8: Validation - Testing Retrieval ===")

        test_queries = [
            "Who works on Project Phoenix?",
            "What does GDPR say about data retention?",
            "Alice Johnson manager"
        ]

        for query in test_queries:
            print(f"\n  Query: '{query}'")

            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            # Search triples
            triple_distances, triple_indices = self.triple_index.search(query_embedding, k=3)
            print(f"    Top 3 matching triples:")
            for i, (idx, score) in enumerate(zip(triple_indices[0], triple_distances[0])):
                metadata = self.triple_metadata[idx]
                print(f"      {i + 1}. {metadata['natural_text']} (score: {score:.3f}, conf: {metadata['confidence']})")

            # Search chunks
            chunk_distances, chunk_indices = self.chunk_index.search(query_embedding, k=3)
            print(f"    Top 3 matching chunks:")
            for i, (idx, score) in enumerate(zip(chunk_indices[0], chunk_distances[0])):
                metadata = self.chunk_metadata[idx]
                preview = metadata['text'][:100] + "..." if len(metadata['text']) > 100 else metadata['text']
                print(f"      {i + 1}. {metadata['doc_filename']} (score: {score:.3f})")
                print(f"         {preview}")

    def run(self):
        """Execute full embedding pipeline"""
        print("=" * 60)
        print("Phase 8: Embeddings & FAISS Indexing")
        print("=" * 60)

        self.connect_neo4j()
        self.load_model()

        self.load_triples()
        self.load_documents()
        self.chunk_documents()

        self.generate_triple_embeddings()
        self.generate_chunk_embeddings()

        self.build_faiss_indexes()
        self.save_outputs()
        self.validate_retrieval()

        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Embedding dimension: {self.stats['embedding_dimension']}")
        print(f"Triples embedded: {self.stats['total_triples']}")
        print(f"Chunks embedded: {self.stats['total_chunks']}")
        print(f"FAISS indexes: 2 (triples + chunks)")
        print(f"\nOutputs saved to: {self.output_dir}")
        print("\nNext: Phase 9 - RAG Pipeline")
        print("=" * 60)

        # Close connections
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("Phase 8: Embeddings & FAISS Indexing")
    print("=" * 60)

    # Load configs
    with open("config/pipeline.yaml") as f:
        pipeline_config = yaml.safe_load(f)

    with open("config/neo4j.yaml") as f:
        neo4j_config = yaml.safe_load(f)

    # Get phase-specific settings
    phase_config = pipeline_config['embedding']

    print(f"Configuration:")
    print(f"  Neo4j URI: {neo4j_config['uri']}")
    print(f"  Model: {phase_config['model_name']}")
    print(f"  Chunk size: {phase_config['chunking']['chunk_size']}")
    print(f"  Chunk overlap: {phase_config['chunking']['chunk_overlap']}")
    print()

    # Initialize with config values
    generator = EmbeddingGenerator(
        neo4j_uri=neo4j_config['uri'],
        neo4j_user=neo4j_config['user'],
        neo4j_password=neo4j_config['password'],
        model_name=phase_config['model_name'],
        chunk_size=phase_config['chunking']['chunk_size'],
        chunk_overlap=phase_config['chunking']['chunk_overlap']
    )

    generator.run()