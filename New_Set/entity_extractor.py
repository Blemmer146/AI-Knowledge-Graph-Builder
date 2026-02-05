"""
Phase 7: Entity & Relationship Extraction (Fixed Resolution)
Handles external entities gracefully - creates ExternalEntity nodes for unresolvable mentions

Author: Enterprise KG Project
Date: 2025-01-25
"""

import yaml
import json
import spacy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import argparse
import sys
from difflib import SequenceMatcher

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed. Run: pip install neo4j")
    sys.exit(1)


class EntityExtractor:
    """Entity extraction with external entity handling"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 spacy_model: str = "en_core_web_md",
                 resolution_threshold: float = 0.90,
                 cooccur_threshold: int = 3,
                 external_entity_threshold: int = 2):
        """
        Initialize entity extraction pipeline

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            spacy_model: spaCy model (sm/md/lg)
            resolution_threshold: Minimum similarity for entity resolution (0.90 = strict)
            cooccur_threshold: Minimum co-occurrences for relationship inference (3+)
            external_entity_threshold: Min mentions to create ExternalEntity (2+ = avoid noise)
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.resolution_threshold = resolution_threshold
        self.cooccur_threshold = cooccur_threshold
        self.external_entity_threshold = external_entity_threshold

        self.driver = None
        self.nlp = None
        self.config = None

        # Load entities registry
        self.entities = {}
        self.entity_lookup = {}

        # Statistics
        self.stats = defaultdict(int)

        # Tracking
        self.extracted_mentions = []
        self.resolved_mentions = []
        self.external_entity_cache = {}
        self.cooccurrences = defaultdict(lambda: defaultdict(int))
        self.triples = []

        # Output paths
        self.data_dir = Path("data")
        self.output_dir = self.data_dir / "knowledge"
        self.output_dir.mkdir(exist_ok=True)

        print(f"Entity Extraction Configuration:")
        print(f"  spaCy model: {spacy_model}")
        print(
            f"  Resolution threshold: {resolution_threshold} ({'strict' if resolution_threshold >= 0.9 else 'loose'})")
        print(f"  Co-occurrence threshold: {cooccur_threshold}+ mentions")
        print(f"  External entity threshold: {external_entity_threshold}+ mentions")

        # Load spaCy model
        try:
            print(f"\nLoading spaCy model '{spacy_model}'...")
            self.nlp = spacy.load(spacy_model)
            print("spaCy model loaded")
        except OSError:
            print(f"\nERROR: spaCy model '{spacy_model}' not found.")
            print(f"Install with: python -m spacy download {spacy_model}")
            sys.exit(1)


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

    def load_entities_registry(self):
        """Load entities.json for canonical entity lookup"""
        print("\nLoading entities registry...")

        entities_path = self.data_dir / "processed/entities.json"  # CHANGED PATH
        with open(entities_path) as f:
            entities_data = json.load(f)

        # Build lookup dictionary
        for emp in entities_data.get('employees', []):
            emp_id = emp['id']
            self.entities[emp_id] = emp
            self.entity_lookup[emp['full_name'].lower()] = emp_id
            self.entity_lookup[emp['first_name'].lower()] = emp_id
            self.entity_lookup[emp['last_name'].lower()] = emp_id
            self.entity_lookup[f"mr. {emp['last_name'].lower()}"] = emp_id
            self.entity_lookup[f"ms. {emp['last_name'].lower()}"] = emp_id

        for proj in entities_data.get('projects', []):
            proj_id = proj['id']
            self.entities[proj_id] = proj
            self.entity_lookup[proj['name'].lower()] = proj_id
            self.entity_lookup[f"project {proj['name'].lower()}"] = proj_id

        for prod in entities_data.get('products', []):
            prod_id = prod['id']
            self.entities[prod_id] = prod
            self.entity_lookup[prod['name'].lower()] = prod_id

        for pol in entities_data.get('policies', []):
            pol_id = pol['id']
            self.entities[pol_id] = pol
            self.entity_lookup[pol['name'].lower()] = pol_id

        for reg in entities_data.get('regulations', []):
            reg_id = reg['id']
            self.entities[reg_id] = reg
            self.entity_lookup[reg['name'].lower()] = reg_id
            if 'full_name' in reg:
                self.entity_lookup[reg['full_name'].lower()] = reg_id

        print(f"Loaded {len(self.entities)} canonical entities")
        print(f"Built lookup with {len(self.entity_lookup)} name variations")

    def extract_entities_from_documents(self):
        """Extract entities from all documents in Neo4j using spaCy NER"""
        print("\n=== STEP 1: Extracting Entities from Documents ===")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                WHERE d.word_count > 0
                RETURN d.id AS doc_id, d.filename AS filename, 
                       d.full_text AS text, d.type AS doc_type
            """)
            documents = [dict(record) for record in result]

        print(f"Processing {len(documents)} documents...")

        for doc in documents:
            doc_id = doc['doc_id']
            text = doc['text']
            spacy_doc = self.nlp(text)

            doc_entities = []
            for ent in spacy_doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'LAW']:
                    mention = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'doc_id': doc_id,
                        'doc_filename': doc['filename'],
                        'doc_type': doc['doc_type']
                    }
                    doc_entities.append(mention)
                    self.extracted_mentions.append(mention)

            self.stats[f'entities_extracted_{doc["doc_type"]}'] += len(doc_entities)
            print(f"  {doc['filename']}: {len(doc_entities)} entities extracted")

        total_extracted = len(self.extracted_mentions)
        self.stats['total_entities_extracted'] = total_extracted
        print(f"\nTotal entities extracted: {total_extracted}")

    def resolve_entities(self):
        """Resolve extracted mentions - canonical or external entities"""
        print("\n=== STEP 2: Resolving Entities ===")

        # First pass: count external entity mentions
        external_mentions = defaultdict(list)

        for mention in self.extracted_mentions:
            mention_text = mention['text'].lower().strip()

            # Try exact match
            if mention_text in self.entity_lookup:
                resolved_id = self.entity_lookup[mention_text]
                self.resolved_mentions.append({
                    **mention,
                    'resolved_id': resolved_id,
                    'resolution_method': 'exact_match',
                    'resolution_type': 'canonical',
                    'confidence': 0.95
                })
                self.stats['exact_matches'] += 1
                continue

            # Try fuzzy match
            best_match, best_score = self._fuzzy_match(mention_text)

            if best_match and best_score >= self.resolution_threshold:
                resolved_id = self.entity_lookup[best_match]
                self.resolved_mentions.append({
                    **mention,
                    'resolved_id': resolved_id,
                    'resolution_method': 'fuzzy_match',
                    'resolution_type': 'canonical',
                    'confidence': round(best_score, 2)
                })
                self.stats['fuzzy_matches'] += 1
            else:
                # Track for external entity creation
                external_mentions[mention_text].append(mention)

        # Second pass: create external entities for mentions >= threshold
        print(f"\n  Processing {len(external_mentions)} unique unresolved entities...")

        for entity_text, mentions in external_mentions.items():
            mention_count = len(mentions)

            if mention_count >= self.external_entity_threshold:
                # Create external entity
                external_id = self._get_or_create_external_entity(
                    entity_text,
                    mentions[0]['label'],
                    mentions[0]['doc_id']
                )

                # Resolve all mentions to this external entity
                for mention in mentions:
                    self.resolved_mentions.append({
                        **mention,
                        'resolved_id': external_id,
                        'resolution_method': 'external_entity',
                        'resolution_type': 'external',
                        'confidence': 0.4,
                        'mention_count': mention_count
                    })
                    self.stats['external_entity_resolved'] += 1
            else:
                # Ignore low-frequency mentions (noise)
                self.stats['ignored_noise'] += len(mentions)

        canonical_count = self.stats['exact_matches'] + self.stats['fuzzy_matches']
        print(f"\nCanonical entities: {canonical_count}")
        print(f"External entities: {self.stats['external_entity_resolved']}")
        print(f"Ignored (noise): {self.stats['ignored_noise']}")
        print(f"Total resolved: {len(self.resolved_mentions)}")
        print(f"  Resolution rate: {len(self.resolved_mentions) / len(self.extracted_mentions) * 100:.1f}%")

    def _fuzzy_match(self, mention_text: str) -> Tuple[Optional[str], float]:
        """Fuzzy match mention to entity lookup"""
        best_match = None
        best_score = 0.0

        for entity_name in self.entity_lookup.keys():
            score = SequenceMatcher(None, mention_text, entity_name).ratio()
            if score > best_score:
                best_score = score
                best_match = entity_name

        return best_match, best_score

    def _get_or_create_external_entity(self, entity_text: str, entity_type: str, first_doc_id: str) -> str:
        """Get or create ExternalEntity node"""
        # Check cache
        if entity_text in self.external_entity_cache:
            return self.external_entity_cache[entity_text]

        # Generate deterministic ID
        external_id = f"ext_{entity_type.lower()}_{abs(hash(entity_text)) % 100000:05d}"

        with self.driver.session() as session:
            session.run("""
                MERGE (e:ExternalEntity {id: $id})
                ON CREATE SET
                    e.name = $name,
                    e.type = $type,
                    e.source = 'phase7_extraction',
                    e.confidence = 0.4,
                    e.first_mentioned = $doc_id,
                    e.created = datetime()
            """, id=external_id, name=entity_text, type=entity_type, doc_id=first_doc_id)

        self.external_entity_cache[entity_text] = external_id
        self.stats['external_entities_created'] += 1
        return external_id

    def create_mention_edges(self):
        """Create MENTIONS edges in Neo4j"""
        print("\n=== STEP 3: Creating MENTIONS Edges ===")

        canonical_mentions = 0
        external_mentions = 0

        with self.driver.session() as session:
            for mention in self.resolved_mentions:
                doc_id = mention['doc_id']
                entity_id = mention['resolved_id']
                mention_text = mention['text']
                confidence = mention['confidence']
                position = mention['start']
                resolution_type = mention['resolution_type']

                # Create MENTIONS edge
                query = """
                MATCH (d:Document {id: $doc_id})
                MATCH (e {id: $entity_id})
                MERGE (d)-[m:MENTIONS {mention_text: $text}]->(e)
                ON CREATE SET 
                    m.confidence = $confidence,
                    m.position = $position,
                    m.extraction_method = 'spacy_ner',
                    m.resolution_type = $resolution_type,
                    m.source = 'phase7_extraction'
                """
                session.run(query,
                            doc_id=doc_id,
                            entity_id=entity_id,
                            text=mention_text,
                            confidence=confidence,
                            position=position,
                            resolution_type=resolution_type)

                if resolution_type == 'canonical':
                    canonical_mentions += 1
                else:
                    external_mentions += 1

        print(f"Created {canonical_mentions} canonical MENTIONS edges")
        print(f"Created {external_mentions} external MENTIONS edges")
        self.stats['mention_edges_created'] = canonical_mentions + external_mentions

    def track_cooccurrences(self):
        """Track entity co-occurrences (canonical entities only)"""
        print("\n=== STEP 4: Tracking Co-occurrences ===")

        # Group CANONICAL mentions by document
        doc_mentions = defaultdict(list)
        for mention in self.resolved_mentions:
            if mention['resolution_type'] == 'canonical':  # Only canonical
                doc_mentions[mention['doc_id']].append(mention['resolved_id'])

        # Count co-occurrences
        for doc_id, entity_ids in doc_mentions.items():
            unique_entities = list(set(entity_ids))

            for i, e1 in enumerate(unique_entities):
                for e2 in unique_entities[i + 1:]:
                    self.cooccurrences[e1][e2] += 1
                    self.cooccurrences[e2][e1] += 1

        # Count significant co-occurrences
        significant = sum(1 for e1 in self.cooccurrences.values()
                         for count in e1.values() if count >= self.cooccur_threshold)

        print(f"Tracked co-occurrences for {len(self.cooccurrences)} canonical entities")
        print(f"Found {significant // 2} significant pairs (>= {self.cooccur_threshold} times)")

    def infer_relationships(self):
        """Infer relationships from co-occurrence patterns"""
        print("\n=== STEP 5: Inferring Relationships ===")

        inferred_count = 0

        with self.driver.session() as session:
            for e1_id, partners in self.cooccurrences.items():
                for e2_id, count in partners.items():
                    if count < self.cooccur_threshold:
                        continue

                    e1_type = e1_id.split('_')[0]
                    e2_type = e2_id.split('_')[0]

                    rel_type, confidence = self._infer_relationship_type(e1_type, e2_type, count)

                    if rel_type:
                        exists = session.run(f"""
                            MATCH (a {{id: $e1}})-[r:{rel_type}]->(b {{id: $e2}})
                            RETURN count(r) AS count
                        """, e1=e1_id, e2=e2_id).single()['count']

                        if exists == 0:
                            query = f"""
                            MATCH (a {{id: $e1}})
                            MATCH (b {{id: $e2}})
                            CREATE (a)-[r:{rel_type} {{
                                source: 'phase7_inference',
                                confidence: $confidence,
                                extraction_method: 'co_occurrence',
                                cooccurrence_count: $count,
                                flagged: false,
                                inferred: true
                            }}]->(b)
                            """
                            session.run(query, e1=e1_id, e2=e2_id,
                                        confidence=confidence, count=count)

                            inferred_count += 1
                            self.stats[f'inferred_{rel_type}'] += 1

        self.stats['total_inferred_relationships'] = inferred_count
        print(f"Inferred {inferred_count} relationships")

        for key, value in self.stats.items():
            if key.startswith('inferred_') and key != 'total_inferred_relationships':
                print(f"  - {key.replace('inferred_', '')}: {value}")

    def _infer_relationship_type(self, type1: str, type2: str, count: int) -> Tuple[Optional[str], float]:
        """Determine relationship type and confidence"""
        if type1 == 'emp' and type2 == 'proj':
            if count >= 5:
                return 'WORKS_ON', 0.7
            elif count >= 3:
                return 'WORKS_ON', 0.6
        elif type1 == 'proj' and type2 == 'prod':
            if count >= 3:
                return 'USES', 0.6
        elif type1 == 'proj' and type2 == 'pol':
            if count >= 2:
                return 'GOVERNED_BY', 0.5
        elif type1 == 'pol' and type2 == 'reg':
            if count >= 2:
                return 'REFERENCES', 0.5
        elif type1 == 'prod' and type2 == 'reg':
            if count >= 2:
                return 'COMPLIES_WITH', 0.5

        return None, 0.0

    def generate_triples(self):
        """Generate triples from Neo4j graph"""
        print("\n=== STEP 6: Generating Triples ===")

        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE NOT type(r) IN ['MENTIONS', 'SENT', 'SENT_TO']
                RETURN a.id AS subject_id,
                       coalesce(a.full_name, a.name) AS subject_name,
                       labels(a)[0] AS subject_type,
                       type(r) AS predicate,
                       b.id AS object_id,
                       coalesce(b.full_name, b.name) AS object_name,
                       labels(b)[0] AS object_type,
                       r.confidence AS confidence,
                       r.source AS source,
                       coalesce(r.flagged, false) AS flagged,
                       coalesce(r.inferred, false) AS inferred
            """)

            for record in result:
                triple = {
                    'subject': {
                        'id': record['subject_id'],
                        'name': record['subject_name'],
                        'type': record['subject_type']
                    },
                    'predicate': record['predicate'],
                    'object': {
                        'id': record['object_id'],
                        'name': record['object_name'],
                        'type': record['object_type']
                    },
                    'confidence': record['confidence'],
                    'source': record['source'],
                    'flagged': record['flagged'],
                    'inferred': record['inferred'],
                    'text': f"{record['subject_name']} {record['predicate'].replace('_', ' ').lower()} {record['object_name']}"
                }
                self.triples.append(triple)

        print(f"Generated {len(self.triples)} triples")

        triples_path = self.output_dir / "triples.json"
        with open(triples_path, 'w') as f:
            json.dump(self.triples, f, indent=2)
        print(f"Saved to {triples_path}")

    def save_extraction_results(self):
        """Save extraction metadata and statistics"""
        print("\n=== STEP 7: Saving Extraction Results ===")

        resolved_path = self.output_dir / "resolved_mentions.json"
        with open(resolved_path, 'w') as f:
            json.dump(self.resolved_mentions, f, indent=2)
        print(f"Saved {len(self.resolved_mentions)} resolved mentions")

        extraction_log = {
            'extraction_timestamp': datetime.now().isoformat(),
            'configuration': {
                'spacy_model': 'en_core_web_md',
                'resolution_threshold': self.resolution_threshold,
                'cooccurrence_threshold': self.cooccur_threshold,
                'external_entity_threshold': self.external_entity_threshold
            },
            'statistics': dict(self.stats),
            'entity_counts': {
                'total_extracted': self.stats['total_entities_extracted'],
                'canonical_resolved': self.stats['exact_matches'] + self.stats['fuzzy_matches'],
                'external_resolved': self.stats['external_entity_resolved'],
                'ignored_noise': self.stats['ignored_noise'],
                'external_entities_created': self.stats.get('external_entities_created', 0),
                'total_resolved': len(self.resolved_mentions),
                'resolution_rate': round(len(self.resolved_mentions) / self.stats['total_entities_extracted'] * 100, 2)
            },
            'relationship_inference': {
                'total_inferred': self.stats.get('total_inferred_relationships', 0),
                'by_type': {k.replace('inferred_', ''): v for k, v in self.stats.items() if k.startswith('inferred_')}
            },
            'triples': {
                'total_count': len(self.triples),
                'ground_truth': len([t for t in self.triples if t.get('confidence') == 1.0]),
                'inferred': len([t for t in self.triples if t.get('inferred') == True]),
                'flagged': len([t for t in self.triples if t.get('flagged') == True])
            }
        }

        log_path = self.output_dir / "extraction_log.json"
        with open(log_path, 'w') as f:
            json.dump(extraction_log, f, indent=2)
        print(f"Saved extraction log to {log_path}")

    def validate_extraction(self):
        """Validate extraction results"""
        print("\n=== STEP 8: Validation ===")

        with self.driver.session() as session:
            mentions_count = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS count").single()['count']

            canonical_nodes = session.run("""
                MATCH (n) 
                WHERE NOT 'ExternalEntity' IN labels(n) AND NOT 'Document' IN labels(n)
                RETURN count(n) AS count
            """).single()['count']

            external_nodes = session.run("MATCH (e:ExternalEntity) RETURN count(e) AS count").single()['count']

            inferred_count = session.run("MATCH ()-[r {inferred: true}]->() RETURN count(r) AS count").single()['count']

            total_rels = session.run("""
                MATCH ()-[r]->()
                WHERE NOT type(r) IN ['MENTIONS', 'SENT', 'SENT_TO']
                RETURN count(r) AS count
            """).single()['count']

        print(f"Total MENTIONS edges: {mentions_count}")
        print(f"Canonical entity nodes: {canonical_nodes}")
        print(f"External entity nodes: {external_nodes}")
        print(f"Inferred relationships: {inferred_count}")
        print(f"Total semantic relationships: {total_rels}")
        print(f"Triples generated: {len(self.triples)}")

    def run(self):
        """Execute full extraction pipeline"""
        print("=" * 60)
        print("Phase 7: Entity & Relationship Extraction (v2)")
        print("=" * 60)

        self.connect_neo4j()
        self.load_entities_registry()

        self.extract_entities_from_documents()
        self.resolve_entities()
        self.create_mention_edges()
        self.track_cooccurrences()
        self.infer_relationships()
        self.generate_triples()
        self.save_extraction_results()
        self.validate_extraction()

        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Entities extracted: {self.stats['total_entities_extracted']}")
        print(f"Canonical resolved: {self.stats['exact_matches'] + self.stats['fuzzy_matches']}")
        print(f"External entities: {self.stats.get('external_entities_created', 0)} created")
        print(f"Resolution rate: {len(self.resolved_mentions) / self.stats['total_entities_extracted'] * 100:.1f}%")
        print(f"Relationships inferred: {self.stats.get('total_inferred_relationships', 0)}")
        print(f"Triples generated: {len(self.triples)}")
        print(f"\nOutputs saved to: {self.output_dir}")
        print("\nNext: Phase 8 - Embeddings & FAISS Indexing")
        print("=" * 60)


if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("Phase 7: Entity Extraction")
    print("=" * 60)

    # Load configs
    with open("config/pipeline.yaml") as f:
        pipeline_config = yaml.safe_load(f)

    with open("config/neo4j.yaml") as f:
        neo4j_config = yaml.safe_load(f)

    # Get phase-specific settings
    phase_config = pipeline_config['extraction']

    print(f"Configuration:")
    print(f"  Neo4j URI: {neo4j_config['uri']}")
    print(f"  spaCy model: {phase_config['spacy_model']}")
    print(f"  Resolution threshold: {phase_config['resolution_threshold']}")
    print(f"  Co-occurrence threshold: {phase_config['cooccurrence_threshold']}")
    print()

    # Initialize with config values
    extractor = EntityExtractor(
        neo4j_uri=neo4j_config['uri'],
        neo4j_user=neo4j_config['user'],
        neo4j_password=neo4j_config['password'],
        spacy_model=phase_config['spacy_model'],
        resolution_threshold=phase_config['resolution_threshold'],
        cooccur_threshold=phase_config['cooccurrence_threshold'],
        external_entity_threshold=phase_config['external_entity_threshold']
    )

    extractor.run()