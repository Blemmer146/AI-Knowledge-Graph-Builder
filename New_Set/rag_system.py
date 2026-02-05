"""
Phase 9: RAG Pipeline - FIXED VERSION
Professional, accurate answers with improved retrieval

Key Fixes:
- Professional tone (not overly chatty)
- Better entity extraction from queries
- Direct Neo4j queries for basic facts (department, role, manager)
- Improved prompt engineering
- Stricter answer formatting

Author: Enterprise KG Project
Date: 2026-02-02 (Fixed)
"""

import yaml
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time
from datetime import datetime
import sys
import re

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    sys.exit(1)


class RAGPipeline:
    """Fixed RAG pipeline with professional responses"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 ollama_model: str = "llama3.2:3b",
                 ollama_base_url: str = "http://localhost:11434",
                 triple_top_k: int = 10, chunk_top_k: int = 15,
                 similarity_threshold: float = 0.15,
                 min_sources_threshold: int = 1):
        """Initialize RAG pipeline"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url

        # Paths
        self.data_dir = Path("data")
        self.embeddings_dir = self.data_dir / "embeddings"

        # Config
        self.config = {
            'retrieval': {
                'triple_top_k': triple_top_k,
                'chunk_top_k': chunk_top_k,
                'similarity_threshold': similarity_threshold,
                'min_sources_threshold': min_sources_threshold
            },
            'ollama': {
                'base_url': ollama_base_url,
                'temperature': 0.0,  # Deterministic for factual answers
                'num_ctx': 4096,
                'num_predict': 300
            }
        }

        # Components
        self.driver = None
        self.model = None
        self.triple_index = None
        self.chunk_index = None
        self.triple_metadata = []
        self.chunk_metadata = []

        # Stats
        self.session_stats = {
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'fallback_triggered': 0,
            'no_sources_found': 0,
            'avg_latency_ms': 0.0,
            'total_latency_ms': 0.0
        }

    def initialize(self):
        """Load all components"""
        print("\n=== Initializing RAG Pipeline ===")

        # Connect Neo4j
        print("Connecting to Neo4j...")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        print("Neo4j connected")

        # Load SentenceTransformer
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded")

        # Load FAISS indexes
        print("Loading FAISS indexes...")
        triple_index_path = self.embeddings_dir / "faiss_triples.index"
        chunk_index_path = self.embeddings_dir / "faiss_chunks.index"

        if not triple_index_path.exists() or not chunk_index_path.exists():
            print(f"ERROR: FAISS indexes not found in {self.embeddings_dir}")
            print("Run Phase 8 first")
            sys.exit(1)

        self.triple_index = faiss.read_index(str(triple_index_path))
        self.chunk_index = faiss.read_index(str(chunk_index_path))
        print(f"Loaded triple index ({self.triple_index.ntotal} vectors)")
        print(f"Loaded chunk index ({self.chunk_index.ntotal} vectors)")

        # Load metadata
        print("Loading metadata...")
        with open(self.embeddings_dir / "triple_metadata.json") as f:
            self.triple_metadata = json.load(f)
        with open(self.embeddings_dir / "chunk_metadata.json") as f:
            self.chunk_metadata = json.load(f)
        print(f"Loaded {len(self.triple_metadata)} triple metadata")
        print(f"Loaded {len(self.chunk_metadata)} chunk metadata")

        # Test Ollama
        print(f"Testing Ollama connection ({self.ollama_model})...")
        if not self._test_ollama():
            print("ERROR: Cannot connect to Ollama")
            print(f"Make sure Ollama is running: ollama serve")
            print(f"And model is pulled: ollama pull {self.ollama_model}")
            sys.exit(1)
        print("Ollama connected")

        print("\nRAG Pipeline ready\n")

    def _test_ollama(self) -> bool:
        """Test Ollama connection"""
        try:
            response = requests.post(
                f"{self.config['ollama']['base_url']}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": "test",
                    "stream": False
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def query(self, question: str, verbose: bool = True) -> dict:
        """Main RAG query with professional responses"""
        start_time = time.time()

        # Initialize metrics
        metrics = {
            'query_start_time': datetime.now().isoformat(),
            'latency_breakdown': {},
            'retrieval_quality': {},
            'reasoning_trace': []
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"QUERY: {question}")
            print('=' * 60)

        # STEP 1: Check if this is a basic fact query (department, role, manager)
        step_start = time.time()
        basic_fact_result = self._try_basic_fact_query(question, verbose=verbose)
        metrics['latency_breakdown']['basic_fact_check_ms'] = (time.time() - step_start) * 1000

        if basic_fact_result:
            # Found answer via direct Neo4j query
            metrics['reasoning_trace'].append({
                'step': 'basic_fact_query',
                'result': 'Answered via direct graph lookup'
            })
            return self._format_basic_fact_response(
                question, basic_fact_result, metrics, start_time
            )

        # STEP 2: Fall back to semantic retrieval
        step_start = time.time()
        triples, chunks = self._retrieve_semantic(question, verbose=verbose)
        metrics['latency_breakdown']['retrieval_ms'] = (time.time() - step_start) * 1000
        metrics['reasoning_trace'].append({
            'step': 'semantic_retrieval',
            'result': f"Found {len(triples)} triples, {len(chunks)} chunks"
        })

        # STEP 3: Check minimum sources
        total_sources = len(triples) + len(chunks)
        min_sources = self.config['retrieval']['min_sources_threshold']

        if total_sources == 0:
            return self._generate_no_source_response(question, metrics, start_time)

        # STEP 4: Get graph context
        step_start = time.time()
        entity_ids = self._extract_entity_ids(triples)
        graph_context = self._retrieve_graph_context(entity_ids, verbose=verbose)
        metrics['latency_breakdown']['graph_context_ms'] = (time.time() - step_start) * 1000

        # STEP 5: Detect contradictions
        step_start = time.time()
        contradictions = self._detect_contradictions(triples, chunks)
        metrics['latency_breakdown']['contradiction_detection_ms'] = (time.time() - step_start) * 1000

        # STEP 6: Build professional prompt
        step_start = time.time()
        prompt = self._build_professional_prompt(
            question, triples, chunks, graph_context, contradictions
        )
        metrics['latency_breakdown']['prompt_building_ms'] = (time.time() - step_start) * 1000

        # STEP 7: Generate answer
        step_start = time.time()
        if verbose:
            print("\nGenerating answer...")
        answer = self._generate_answer(prompt)
        metrics['latency_breakdown']['llm_generation_ms'] = (time.time() - step_start) * 1000

        # STEP 8: Calculate confidence
        confidence = self._calculate_confidence(
            question, triples, chunks, contradictions, answer
        )

        # Build response
        total_time = time.time() - start_time
        metrics['total_latency_ms'] = total_time * 1000
        metrics['query_end_time'] = datetime.now().isoformat()

        metrics['retrieval_quality'] = {
            'triples_found': len(triples),
            'chunks_found': len(chunks),
            'total_sources': total_sources,
            'has_sufficient_sources': total_sources >= min_sources,
            'avg_triple_confidence': float(np.mean([t.get('confidence', 0.5) for t in triples])) if triples else 0.0,
            'avg_similarity_score': float(np.mean([t.get('similarity_score', 0.0) for t in triples] +
                                                   [c.get('similarity_score', 0.0) for c in chunks])) if (triples or chunks) else 0.0
        }

        # Update session stats
        self.session_stats['queries_processed'] += 1
        self.session_stats['successful_queries'] += 1
        self.session_stats['total_latency_ms'] += metrics['total_latency_ms']
        self.session_stats['avg_latency_ms'] = (
            self.session_stats['total_latency_ms'] / self.session_stats['queries_processed']
        )

        response = {
            'question': question,
            'answer': answer,
            'sources': {
                'triples': triples,
                'chunks': chunks[:5],
                'graph_context': graph_context,
                'contradictions': contradictions,
                'source_manifest': self._build_source_manifest(triples, chunks)
            },
            'stats': {
                'triples_retrieved': len(triples),
                'chunks_retrieved': len(chunks),
                'entities_in_context': len(entity_ids),
                'contradictions_found': len(contradictions),
                'fallback_triggered': False
            },
            'metrics': metrics,
            'status': 'success',
            'confidence': confidence
        }

        if verbose:
            self._print_response(response)

        return response

    def _try_basic_fact_query(self, question: str, verbose: bool = False) -> Optional[dict]:
        """Try to answer basic fact queries directly from Neo4j"""
        question_lower = question.lower()

        # Extract person name
        person_name = self._extract_person_name(question)
        if not person_name:
            return None

        # Determine query type
        query_type = None
        if any(word in question_lower for word in ['department', 'dept']):
            query_type = 'department'
        elif any(word in question_lower for word in ['role', 'position', 'job', 'title']):
            query_type = 'role'
        elif any(word in question_lower for word in ['manager', 'supervisor', 'reports to', 'boss']):
            query_type = 'manager'

        if not query_type:
            return None

        if verbose:
            print(f"\nDetected basic fact query: {query_type} for {person_name}")

        # Query Neo4j directly
        with self.driver.session() as session:
            # Find employee by name (fuzzy match on first/last name)
            name_parts = person_name.split()

            if query_type == 'department':
                result = session.run("""
                    MATCH (e:Employee)
                    WHERE e.first_name CONTAINS $first_name 
                       OR e.last_name CONTAINS $last_name
                       OR e.full_name CONTAINS $full_name
                    RETURN e.department AS department, e.full_name AS name
                    LIMIT 1
                """, first_name=name_parts[0] if name_parts else person_name,
                     last_name=name_parts[-1] if len(name_parts) > 1 else person_name,
                     full_name=person_name)

                record = result.single()
                if record and record['department']:
                    return {
                        'type': 'department',
                        'value': record['department'],
                        'person': record['name'],
                        'confidence': 1.0,
                        'source': 'employees.csv'
                    }

            elif query_type == 'role':
                result = session.run("""
                    MATCH (e:Employee)
                    WHERE e.first_name CONTAINS $first_name 
                       OR e.last_name CONTAINS $last_name
                       OR e.full_name CONTAINS $full_name
                    RETURN e.role AS role, e.full_name AS name
                    LIMIT 1
                """, first_name=name_parts[0] if name_parts else person_name,
                     last_name=name_parts[-1] if len(name_parts) > 1 else person_name,
                     full_name=person_name)

                record = result.single()
                if record and record['role']:
                    return {
                        'type': 'role',
                        'value': record['role'],
                        'person': record['name'],
                        'confidence': 1.0,
                        'source': 'employees.csv'
                    }

            elif query_type == 'manager':
                result = session.run("""
                    MATCH (e:Employee)-[:REPORTS_TO]->(m:Employee)
                    WHERE e.first_name CONTAINS $first_name 
                       OR e.last_name CONTAINS $last_name
                       OR e.full_name CONTAINS $full_name
                    RETURN m.full_name AS manager, e.full_name AS employee
                    LIMIT 1
                """, first_name=name_parts[0] if name_parts else person_name,
                     last_name=name_parts[-1] if len(name_parts) > 1 else person_name,
                     full_name=person_name)

                record = result.single()
                if record and record['manager']:
                    return {
                        'type': 'manager',
                        'value': record['manager'],
                        'person': record['employee'],
                        'confidence': 1.0,
                        'source': 'employees.csv'
                    }

        return None

    def _format_basic_fact_response(self, question: str, fact_result: dict,
                                    metrics: dict, start_time: float) -> dict:
        """Format response for basic fact queries"""
        fact_type = fact_result['type']
        value = fact_result['value']
        person = fact_result['person']

        # Generate professional answer
        if fact_type == 'department':
            answer = f"{person} works in {value}."
        elif fact_type == 'role':
            answer = f"{person} is a {value}."
        elif fact_type == 'manager':
            answer = f"{person}'s manager is {value}."
        else:
            answer = f"{value}."

        total_time = time.time() - start_time
        metrics['total_latency_ms'] = total_time * 1000
        metrics['query_end_time'] = datetime.now().isoformat()

        # Create synthetic triple for source tracking
        triple = {
            'natural_text': answer,
            'confidence': fact_result['confidence'],
            'source_file': fact_result['source'],
            'subject_id': 'employee',
            'predicate': fact_type,
            'object_id': value
        }

        response = {
            'question': question,
            'answer': answer,
            'sources': {
                'triples': [triple],
                'chunks': [],
                'graph_context': {},
                'contradictions': [],
                'source_manifest': {
                    'files': [fact_result['source']],
                    'file_details': {
                        fact_result['source']: {
                            'used_in_triples': 1,
                            'used_in_chunks': 0
                        }
                    },
                    'triple_sources': [triple],
                    'chunk_sources': []
                }
            },
            'stats': {
                'triples_retrieved': 1,
                'chunks_retrieved': 0,
                'entities_in_context': 1,
                'contradictions_found': 0,
                'fallback_triggered': True
            },
            'metrics': metrics,
            'status': 'success',
            'confidence': fact_result['confidence']
        }

        return response

    def _extract_person_name(self, query: str) -> Optional[str]:
        """Extract person name from query"""
        # Look for capitalized words (names)
        words = query.split()
        name_words = []

        for i, word in enumerate(words):
            # Skip common question words
            if word.lower() in ['who', 'what', 'is', 'the', 'does', 'work', 'in',
                               'manager', 'department', 'role', 'position', 's', "'s"]:
                continue

            # Check if capitalized (likely a name)
            if word and word[0].isupper():
                # Handle possessive (e.g., "Alice's")
                clean_word = word.rstrip("'s")
                name_words.append(clean_word)

        # Return first two capitalized words as name
        if len(name_words) >= 2:
            return ' '.join(name_words[:2])
        elif len(name_words) == 1:
            return name_words[0]

        return None

    def _retrieve_semantic(self, query: str, verbose: bool = False) -> Tuple[List[dict], List[dict]]:
        """Semantic retrieval via FAISS"""
        k_triples = self.config['retrieval']['triple_top_k']
        k_chunks = self.config['retrieval']['chunk_top_k']
        threshold = self.config['retrieval']['similarity_threshold']

        # Encode query
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        # Search triples
        distances, indices = self.triple_index.search(query_emb, k_triples)
        triples = []
        for idx, score in zip(indices[0], distances[0]):
            if score >= threshold and idx < len(self.triple_metadata):
                metadata = self.triple_metadata[idx].copy()
                metadata['similarity_score'] = float(score)
                triples.append(metadata)

        # Search chunks
        distances, indices = self.chunk_index.search(query_emb, k_chunks)
        chunks = []
        for idx, score in zip(indices[0], distances[0]):
            if score >= threshold and idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx].copy()
                metadata['similarity_score'] = float(score)
                chunks.append(metadata)

        if verbose:
            print(f"\nRetrieved {len(triples)} triples, {len(chunks)} chunks")

        return triples, chunks

    def _extract_entity_ids(self, triples: List[dict]) -> List[str]:
        """Extract entity IDs from triples"""
        entity_ids = set()
        for triple in triples:
            if 'subject_id' in triple:
                entity_ids.add(triple['subject_id'])
            if 'object_id' in triple:
                entity_ids.add(triple['object_id'])
            if 'entity_id' in triple:
                entity_ids.add(triple['entity_id'])
        return list(entity_ids)

    def _retrieve_graph_context(self, entity_ids: List[str], verbose: bool = False) -> dict:
        """Retrieve graph context from Neo4j"""
        if not entity_ids:
            return {}

        context = {}

        with self.driver.session() as session:
            for entity_id in entity_ids[:5]:
                result = session.run("""
                    MATCH (e {id: $id})-[r]->(n)
                    WHERE NOT type(r) IN ['MENTIONS', 'SENT', 'SENT_TO']
                    RETURN e.id AS entity_id,
                           coalesce(e.full_name, e.name) AS entity_name,
                           type(r) AS relationship,
                           coalesce(n.full_name, n.name) AS target_name,
                           r.confidence AS confidence
                    LIMIT 10
                """, id=entity_id)

                relationships = []
                entity_name = None
                for record in result:
                    entity_name = record['entity_name']
                    relationships.append({
                        'relationship': record['relationship'],
                        'target': record['target_name'],
                        'confidence': record['confidence']
                    })

                if relationships:
                    context[entity_id] = {
                        'name': entity_name or entity_id,
                        'relationships': relationships
                    }

        return context

    def _detect_contradictions(self, triples: List[dict], chunks: List[dict]) -> List[dict]:
        """Detect contradictions"""
        contradictions = []

        for triple in triples:
            if triple.get('flagged'):
                contradictions.append({
                    'type': 'triple',
                    'text': triple['natural_text'],
                    'confidence': triple['confidence'],
                    'source': triple.get('source_file', triple.get('source', 'unknown')),
                    'reason': 'Flagged as contradictory in knowledge graph',
                    'severity': 'high'
                })

        for chunk in chunks:
            if chunk.get('has_contradictions'):
                contradictions.append({
                    'type': 'chunk',
                    'text': chunk['text'][:200] + "...",
                    'source': chunk['doc_filename'],
                    'confidence': chunk.get('doc_confidence', 0.5),
                    'reason': 'From document flagged with contradictions',
                    'severity': 'medium'
                })

        return contradictions

    def _build_professional_prompt(self, question: str, triples: List[dict],
                                   chunks: List[dict], graph_context: dict,
                                   contradictions: List[dict]) -> str:
        """Build prompt for professional answers"""

        prompt = f"""You are a knowledge assistant for an enterprise organization. Answer questions accurately and professionally based on the provided information.

QUESTION: {question}

INSTRUCTIONS:
1. Provide a direct, factual answer
2. Be concise and professional (2-3 sentences maximum)
3. State facts clearly without hedging unless genuinely uncertain
4. Do not use casual language or filler phrases
5. If contradictions exist, note them briefly

AVAILABLE INFORMATION:

"""

        # High confidence facts
        if triples:
            prompt += "FACTS FROM STRUCTURED DATA:\n"
            for i, t in enumerate(triples[:5], 1):
                prompt += f"{i}. {t['natural_text']} (confidence: {t['confidence']:.2f})\n"
            prompt += "\n"

        # Supporting context
        if chunks:
            prompt += "ADDITIONAL CONTEXT FROM DOCUMENTS:\n"
            for i, c in enumerate(chunks[:3], 1):
                preview = c['text'][:150].replace('\n', ' ')
                prompt += f"{i}. {preview}...\n"
            prompt += "\n"

        # Contradictions
        if contradictions:
            prompt += "NOTE - CONFLICTING INFORMATION:\n"
            for i, contra in enumerate(contradictions[:2], 1):
                prompt += f"{i}. {contra['text'][:100]}... (source: {contra['source']})\n"
            prompt += "Prioritize higher confidence sources.\n\n"

        prompt += """YOUR ANSWER (be direct and professional):"""

        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer via Ollama"""
        try:
            response = requests.post(
                f"{self.config['ollama']['base_url']}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config['ollama']['temperature'],
                        "num_ctx": self.config['ollama']['num_ctx'],
                        "num_predict": self.config['ollama']['num_predict']
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                raw_answer = response.json()['response'].strip()

                # Clean answer
                clean_answer = self._clean_answer(raw_answer)

                return clean_answer
            else:
                return "Unable to process the query at this time."

        except Exception as e:
            return "Unable to process the query at this time."

    def _clean_answer(self, answer: str) -> str:
        """Clean up answer"""
        # Remove excessive newlines
        answer = re.sub(r'\n\s*\n+', '\n', answer)

        # Remove leading/trailing whitespace
        answer = answer.strip()

        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]

        return answer

    def _calculate_confidence(self, question: str, triples: List[dict],
                             chunks: List[dict], contradictions: List[dict],
                             answer: str) -> float:
        """Calculate confidence score"""
        if not triples and not chunks:
            return 0.0

        # Base confidence from sources
        triple_confidences = [t.get('confidence', 0.5) for t in triples]
        similarity_scores = [t.get('similarity_score', 0.5) for t in triples] + \
                           [c.get('similarity_score', 0.5) for c in chunks]

        base_confidence = np.mean(triple_confidences + similarity_scores) if (triple_confidences + similarity_scores) else 0.5

        # Penalty for contradictions
        contradiction_penalty = min(len(contradictions) * 0.15, 0.3)

        # Boost for high-quality sources
        csv_count = sum(1 for t in triples if 'csv' in t.get('source_file', '').lower())
        csv_boost = min(csv_count * 0.15, 0.30)

        confidence = base_confidence - contradiction_penalty + csv_boost

        return max(0.0, min(1.0, round(confidence, 3)))

    def _build_source_manifest(self, triples: List[dict], chunks: List[dict]) -> dict:
        """Build source manifest"""
        manifest = {
            'files': set(),
            'file_details': {},
            'triple_sources': [],
            'chunk_sources': []
        }

        for i, t in enumerate(triples, 1):
            source_file = t.get('source_file', t.get('source', 'unknown'))
            manifest['files'].add(source_file)
            manifest['triple_sources'].append({
                'index': i,
                'text': t['natural_text'],
                'file': source_file,
                'confidence': t['confidence']
            })

        for i, c in enumerate(chunks, 1):
            source_file = c.get('doc_filename', 'unknown')
            manifest['files'].add(source_file)
            manifest['chunk_sources'].append({
                'index': i,
                'file': source_file,
                'chunk_index': c.get('chunk_index', 0),
                'similarity': c['similarity_score']
            })

        manifest['files'] = list(manifest['files'])
        manifest['file_details'] = {
            file: {
                'used_in_triples': sum(1 for t in manifest['triple_sources'] if t['file'] == file),
                'used_in_chunks': sum(1 for c in manifest['chunk_sources'] if c['file'] == file)
            }
            for file in manifest['files']
        }

        return manifest

    def _generate_no_source_response(self, question: str, metrics: dict, start_time: float) -> dict:
        """Generate response when no sources found"""
        self.session_stats['no_sources_found'] += 1
        self.session_stats['failed_queries'] += 1

        total_time = time.time() - start_time
        metrics['total_latency_ms'] = total_time * 1000
        metrics['query_end_time'] = datetime.now().isoformat()

        answer = "I don't have sufficient information to answer this question."

        return {
            'question': question,
            'answer': answer,
            'sources': {
                'triples': [],
                'chunks': [],
                'graph_context': {},
                'contradictions': [],
                'source_manifest': {'files': [], 'file_details': {}}
            },
            'stats': {
                'triples_retrieved': 0,
                'chunks_retrieved': 0,
                'entities_in_context': 0,
                'contradictions_found': 0,
                'fallback_triggered': False
            },
            'metrics': metrics,
            'status': 'no_sources_found',
            'confidence': 0.0
        }

    def _print_response(self, response: dict):
        """Print formatted response"""
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(response['answer'])

        conf = response['confidence']
        conf_label = 'HIGH' if conf >= 0.8 else 'MEDIUM' if conf >= 0.5 else 'LOW'
        print(f"\n📊 Confidence: {conf:.1%} ({conf_label})")

        if response['sources']['contradictions']:
            print("\n⚠️  Note: Some conflicting information was found")

        print("\n" + "-" * 60)
        print(f"Sources: {response['stats']['triples_retrieved']} facts, "
              f"{response['stats']['chunks_retrieved']} documents")
        print(f"Processing time: {response['metrics']['total_latency_ms']:.0f}ms")
        print("=" * 60)

    def get_session_stats(self) -> dict:
        """Get session statistics"""
        return self.session_stats.copy()

    def interactive_mode(self):
        """Interactive query loop"""
        print("\n" + "=" * 60)
        print("RAG PIPELINE - INTERACTIVE MODE")
        print("=" * 60)
        print("Ask questions. Type 'quit' to exit, 'stats' for statistics.")
        print()

        while True:
            try:
                question = input("\nYour question: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    break

                if question.lower() == 'stats':
                    print("\n📊 SESSION STATISTICS:")
                    for key, value in self.session_stats.items():
                        print(f"  {key}: {value}")
                    continue

                self.query(question, verbose=True)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                continue

    def close(self):
        """Close connections"""
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("Phase 9: Enhanced RAG System (Fixed)")
    print("=" * 60)

    with open("config/pipeline.yaml") as f:
        pipeline_config = yaml.safe_load(f)

    with open("config/neo4j.yaml") as f:
        neo4j_config = yaml.safe_load(f)

    with open("config/ollama.yaml") as f:
        ollama_config = yaml.safe_load(f)

    phase_config = pipeline_config['rag']

    rag_pipeline = RAGPipeline(
        neo4j_uri=neo4j_config['uri'],
        neo4j_user=neo4j_config['user'],
        neo4j_password=neo4j_config['password'],
        ollama_model=ollama_config['model'],
        ollama_base_url=ollama_config['base_url'],
        triple_top_k=phase_config['retrieval']['triple_top_k'],
        chunk_top_k=phase_config['retrieval']['chunk_top_k'],
        similarity_threshold=phase_config['retrieval']['similarity_threshold'],
        min_sources_threshold=phase_config['retrieval'].get('min_sources_threshold', 1)
    )

    rag_pipeline.initialize()

    ready_flag = Path("data/rag_ready.flag")
    ready_flag.parent.mkdir(parents=True, exist_ok=True)
    ready_flag.write_text("RAG ready")

    mode = phase_config.get("mode", "auto")

    if mode == "interactive":
        rag_pipeline.interactive_mode()
    elif mode == "non_interactive":
        print("RAG system initialized successfully (non-interactive mode)")
    elif mode == "auto":
        if sys.stdin.isatty():
            rag_pipeline.interactive_mode()
        else:
            print("RAG system initialized successfully (non-interactive mode)")