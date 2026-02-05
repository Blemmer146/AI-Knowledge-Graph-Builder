"""
Phase 6: Data Ingestion Pipeline
Loads all generated data into Neo4j + generates CSV exports
"""

import yaml, json, pandas as pd, argparse, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    import docx
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"ERROR: Missing dependency - {e}")
    print("Run: pip install neo4j python-docx pymupdf")
    sys.exit(1)

class DataIngestor:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 clear_db: bool = False, batch_size: int = 1000):
        """Initialize data ingestor"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.clear_db = clear_db
        self.batch_size = batch_size
        self.driver = None
        self.config = None
        self.stats = defaultdict(int)

        self.data_dir = Path("data")
        self.graph_dir = self.data_dir / "graph"
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.nodes_data = []
        self.edges_data = []

    def load_config(self):
        print(f"Loading config from {self.config_path}...")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        print(f"Config loaded: {self.config['company']['name']}")
        return self.config

    def connect_neo4j(self):
        print(f"\nConnecting to Neo4j at {self.neo4j_uri}...")
        try:
            self.driver = GraphDatabase.driver(self.neo4j_uri,
                                              auth=(self.neo4j_user, self.neo4j_password))
            with self.driver.session() as s:
                s.run("RETURN 1").single()
            print("Neo4j connection successful")
        except ServiceUnavailable:
            print("✗ ERROR: Cannot connect to Neo4j. Is it running?")
            sys.exit(1)
        except AuthError:
            print(f"✗ ERROR: Auth failed. Check user/password.")
            sys.exit(1)

    def clear_database(self):
        if not self.clear_db:
            print("\nSkipping DB clear (use --clear-database to wipe)")
            return
        print("\nCLEARING DATABASE...")
        with self.driver.session() as s:
            s.run("MATCH ()-[r]->() DELETE r")
            s.run("MATCH (n) DELETE n")
        print("Database cleared")

    def create_indexes(self):
        print("\nCreating indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (e:Employee) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Project) ON (p.id)",
            "CREATE INDEX IF NOT EXISTS FOR (prod:Product) ON (prod.id)",
            "CREATE INDEX IF NOT EXISTS FOR (pol:Policy) ON (pol.id)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Regulation) ON (r.id)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:ShadowEntity) ON (s.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Employee) ON (e.email)"
        ]
        with self.driver.session() as s:
            for idx in indexes:
                s.run(idx)
        print(f"Created {len(indexes)} indexes")

    def ingest_entities(self):
        print("\n=== STEP 1: Ingesting Entities ===")
        entities_path = self.data_dir / "processed" / "entities.json"
        with open(entities_path) as f:
            entities = json.load(f)

        with self.driver.session() as s:
            for emp in entities.get('employees', []):
                s.run("""CREATE (e:Employee {
                    id: $id, full_name: $full_name, first_name: $first_name,
                    last_name: $last_name, role: $role, department: $department,
                    email: $email, source: 'employees.csv', confidence: 1.0
                })""", **emp)
                self.stats['employees'] += 1
                self.nodes_data.append({'node_id': emp['id'], 'label': 'Employee',
                                       'properties_json': json.dumps(emp)})

            for proj in entities.get('projects', []):
                s.run("""CREATE (p:Project {
                    id: $id, name: $name, description: $description,
                    status: $status, department: $department,
                    source: 'projects.csv', confidence: 1.0
                })""", **proj)
                self.stats['projects'] += 1
                self.nodes_data.append({'node_id': proj['id'], 'label': 'Project',
                                       'properties_json': json.dumps(proj)})

            for prod in entities.get('products', []):
                s.run("""CREATE (p:Product {
                    id: $id, name: $name, vendor: $vendor, category: $category,
                    source: 'products.csv', confidence: 1.0
                })""", **prod)
                self.stats['products'] += 1
                self.nodes_data.append({'node_id': prod['id'], 'label': 'Product',
                                       'properties_json': json.dumps(prod)})

            for pol in entities.get('policies', []):
                s.run("""CREATE (p:Policy {
                    id: $id, name: $name, category: $category, version: $version,
                    source: 'policies.csv', confidence: 1.0
                })""", **pol)
                self.stats['policies'] += 1
                self.nodes_data.append({'node_id': pol['id'], 'label': 'Policy',
                                       'properties_json': json.dumps(pol)})

            for reg in entities.get('regulations', []):
                s.run("""CREATE (r:Regulation {
                    id: $id, name: $name, full_name: $full_name, type: $type,
                    source: 'entities.json', confidence: 1.0
                })""", **reg)
                self.stats['regulations'] += 1
                self.nodes_data.append({'node_id': reg['id'], 'label': 'Regulation',
                                       'properties_json': json.dumps(reg)})

        print(f"{self.stats['employees']} employees, {self.stats['projects']} projects")
        print(f"{self.stats['products']} products, {self.stats['policies']} policies")
        print(f"{self.stats['regulations']} regulations")

    def ingest_csv_relationships(self):
        print("\n=== STEP 2: Ingesting CSV Relationships ===")

        # REPORTS_TO from employees.csv
        emp_csv = self.data_dir / "source/structured/employees.csv"
        if emp_csv.exists():
            df = pd.read_csv(emp_csv)
            with self.driver.session() as s:
                for _, r in df.iterrows():
                    if pd.notna(r['manager_id']):
                        s.run("""MATCH (e:Employee {id: $emp}), (m:Employee {id: $mgr})
                                CREATE (e)-[:REPORTS_TO {source: 'employees.csv', 
                                confidence: 1.0, since: $date}]->(m)""",
                              emp=r['employee_id'], mgr=r['manager_id'],
                              date=str(r['hire_date']))
                        self.stats['reports_to'] += 1
                        self.edges_data.append({
                            'source_id': r['employee_id'], 'relationship_type': 'REPORTS_TO',
                            'target_id': r['manager_id'],
                            'properties_json': json.dumps({'confidence': 1.0})
                        })
            print(f"{self.stats['reports_to']} REPORTS_TO")

        # WORKS_ON from project_assignments.csv
        assign_csv = self.data_dir / "source/structured/project_assignments.csv"
        if assign_csv.exists():
            df = pd.read_csv(assign_csv)
            with self.driver.session() as s:
                for _, r in df.iterrows():
                    s.run("""MATCH (e:Employee {id: $emp}), (p:Project {id: $proj})
                            CREATE (e)-[:WORKS_ON {source: 'project_assignments.csv',
                            confidence: 1.0, role: $role, allocation_pct: $alloc,
                            start_date: $date}]->(p)""",
                          emp=r['employee_id'], proj=r['project_id'],
                          role=r['role'], alloc=int(r['allocation_pct']),
                          date=str(r['start_date']))
                    self.stats['works_on'] += 1
                    self.edges_data.append({
                        'source_id': r['employee_id'], 'relationship_type': 'WORKS_ON',
                        'target_id': r['project_id'],
                        'properties_json': json.dumps({'confidence': 1.0, 'role': r['role']})
                    })
            print(f"{self.stats['works_on']} WORKS_ON")

        # OWNS from policies.csv
        pol_csv = self.data_dir / "source/structured/policies.csv"
        if pol_csv.exists():
            df = pd.read_csv(pol_csv)
            with self.driver.session() as s:
                for _, r in df.iterrows():
                    if pd.notna(r['owner_id']):
                        s.run("""MATCH (e:Employee {id: $owner}), (p:Policy {id: $pol})
                                CREATE (e)-[:OWNS {source: 'policies.csv',
                                confidence: 1.0, since: $date}]->(p)""",
                              owner=r['owner_id'], pol=r['policy_id'],
                              date=str(r['effective_date']))
                        self.stats['owns'] += 1
                        self.edges_data.append({
                            'source_id': r['owner_id'], 'relationship_type': 'OWNS',
                            'target_id': r['policy_id'],
                            'properties_json': json.dumps({'confidence': 1.0})
                        })
            print(f"{self.stats['owns']} OWNS")

    def extract_text_from_docx(self, path: Path) -> Tuple[str, int]:
        try:
            doc = docx.Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
            return text, len(text.split())
        except Exception as e:
            print(f"  DOCX error {path.name}: {e}")
            return "", 0

    def extract_text_from_txt(self, path: Path) -> Tuple[str, int]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text, len(text.split())
        except Exception as e:
            print(f"  TXT error {path.name}: {e}")
            return "", 0

    def extract_text_from_pdf(self, path: Path) -> Tuple[str, int]:
        try:
            doc = fitz.open(path)
            text = "".join([p.get_text() for p in doc])
            doc.close()
            return text, len(text.split())
        except Exception as e:
            print(f"  PDF error {path.name}: {e}")
            return "", 0

    def ingest_documents(self):
        print("\n=== STEP 3: Ingesting Documents ===")

        meta_path = self.data_dir / "metadata.json"
        metadata = json.load(open(meta_path)) if meta_path.exists() else {'documents': []}
        doc_meta_map = {d['id']: d for d in metadata.get('documents', [])}

        # DOCX reports
        print("Processing DOCX reports...")
        for f in sorted((self.data_dir / "source/semi_structured").glob("*.docx")):
            doc_id = f.stem.replace('project_report_', 'doc_')
            text, wc = self.extract_text_from_docx(f)
            self._create_doc_node(doc_id, f.name, 'semi_structured', 'generated',
                                 text, wc, doc_meta_map.get(doc_id, {}))
            print(f"  {f.name} ({wc} words)")

        # Emails (TXT)
        print("Processing emails...")
        for f in sorted((self.data_dir / "source/unstructured").glob("email_*.txt")):
            text, wc = self.extract_text_from_txt(f)
            self._create_doc_node(f.stem, f.name, 'unstructured', 'generated',
                                 text, wc, doc_meta_map.get(f.stem, {}))
            print(f"  {f.name} ({wc} words)")

        # External docs (PDF/TXT)
        print("Processing external docs...")
        for f in sorted((self.data_dir / "source/external").glob("*")):
            if f.suffix.lower() in ['.pdf', '.txt']:
                doc_id = f"ext_{f.stem.lower()[:10]}"
                text, wc = (self.extract_text_from_pdf(f) if f.suffix == '.pdf'
                           else self.extract_text_from_txt(f))
                self._create_doc_node(doc_id, f.name, 'external', 'downloaded',
                                     text, wc, {})
                print(f"  {f.name} ({wc} words)")

        print(f"Created {self.stats['documents']} documents")

    def _create_doc_node(self, doc_id, fname, dtype, src, text, wc, meta):
        conf = meta.get('confidence_alignment', 1.0)
        ts = meta.get('timestamp', '')
        contras = meta.get('contradictions', [])

        with self.driver.session() as s:
            s.run("""CREATE (d:Document {
                id: $id, filename: $fname, type: $dtype, source: $src,
                timestamp: $ts, full_text: $text, word_count: $wc,
                confidence: $conf, has_contradictions: $has_c,
                contradiction_count: $c_count, ingestion_date: $idate
            })""", id=doc_id, fname=fname, dtype=dtype, src=src, ts=ts,
                 text=text, wc=wc, conf=conf, has_c=len(contras)>0,
                 c_count=len(contras), idate=datetime.now().isoformat())

        self.stats['documents'] += 1
        self.nodes_data.append({
            'node_id': doc_id, 'label': 'Document',
            'properties_json': json.dumps({'filename': fname, 'word_count': wc})
        })

    def ingest_email_metadata(self):
        print("\n=== STEP 4: Ingesting Email Metadata ===")
        em_path = self.data_dir / "processed/emails_metadata.json"
        if not em_path.exists():
            print("emails_metadata.json not found")
            return

        emails = json.load(open(em_path)).get('emails', [])
        with self.driver.session() as s:
            for e in emails:
                from_emp = self._find_emp_by_email(e.get('from_email', ''))
                to_emp = self._find_emp_by_email(e.get('to_email', ''))

                if from_emp:
                    s.run("""MATCH (e:Employee {id: $emp}), (d:Document {id: $doc})
                            CREATE (e)-[:SENT {timestamp: $ts, subject: $subj,
                            source: 'emails_metadata.json'}]->(d)""",
                          emp=from_emp, doc=e['id'], ts=e.get('timestamp'),
                          subj=e.get('subject'))
                    self.stats['sent'] += 1

                if to_emp:
                    s.run("""MATCH (d:Document {id: $doc}), (e:Employee {id: $emp})
                            CREATE (d)-[:SENT_TO {timestamp: $ts,
                            source: 'emails_metadata.json'}]->(e)""",
                          doc=e['id'], emp=to_emp, ts=e.get('timestamp'))
                    self.stats['sent_to'] += 1

        print(f"{self.stats['sent']} SENT, {self.stats['sent_to']} SENT_TO")

    def _find_emp_by_email(self, email: str) -> Optional[str]:
        if not email:
            return None
        with self.driver.session() as s:
            r = s.run("MATCH (e:Employee {email: $email}) RETURN e.id", email=email)
            rec = r.single()
            return rec['e.id'] if rec else None

    def ingest_document_mentions(self):
        print("\n=== STEP 5: Ingesting Document Mentions ===")
        meta_path = self.data_dir / "processed" / "metadata.json"
        if not meta_path.exists():
            print("metadata.json not found")
            return

        metadata = json.load(open(meta_path))
        with self.driver.session() as s:
            for doc in metadata.get('documents', []):
                for m in doc.get('entities_mentioned', []):
                    eid = m.get('id', '')
                    if not self._entity_exists(eid):
                        continue

                    s.run("""MATCH (d:Document {id: $doc}), (e {id: $eid})
                            CREATE (d)-[:MENTIONS {mention_text: $txt,
                            confidence: 0.9, extraction_method: 'metadata',
                            source: 'metadata.json'}]->(e)""",
                          doc=doc['id'], eid=eid, txt=m.get('mention_text'))
                    self.stats['mentions'] += 1

        print(f"{self.stats['mentions']} MENTIONS")

    def _entity_exists(self, eid: str) -> bool:
        with self.driver.session() as s:
            r = s.run("MATCH (n {id: $id}) RETURN count(n) AS c", id=eid)
            return r.single()['c'] > 0

    def ingest_contradictions(self):
        print("\n=== STEP 6: Ingesting Contradictions ===")
        meta = json.load(open(self.data_dir /"processed"/ "metadata.json"))

        with self.driver.session() as s:
            for doc in meta.get('documents', []):
                for c in doc.get('contradictions', []):
                    if c['type'] == 'project_assignment':
                        if not self._entity_exists(c['document_value']):
                            continue
                        s.run("""MATCH (e:Employee {id: $emp}), (p:Project {id: $proj})
                                CREATE (e)-[:WORKS_ON {source: $doc, confidence: 0.3,
                                flagged: true, contradiction_reason: $reason,
                                extraction_method: 'contradiction'}]->(p)""",
                              emp=c['entity'], proj=c['document_value'],
                              doc=doc['id'], reason=c['explanation'])
                        self.stats['contradictions'] += 1

        print(f"{self.stats['contradictions']} contradictions ingested")

    def ingest_shadow_entities(self):
        print("\n=== STEP 7: Ingesting Shadow Entities ===")
        meta = json.load(open(self.data_dir /"processed" / "metadata.json"))

        shadow_entities = {}
        with self.driver.session() as s:
            for doc in meta.get('documents', []):
                for c in doc.get('contradictions', []):
                    if c['type'] in ['product_mention', 'policy_reference']:
                        name = c.get('document_value') or c.get('entity')
                        if name not in shadow_entities:
                            sid = f"shadow_{len(shadow_entities)+1:03d}"
                            stype = 'Product' if c['type'] == 'product_mention' else 'Policy'
                            s.run("""CREATE (s:ShadowEntity {id: $id, name: $name,
                                    type: $type, source: $doc, confidence: 0.3,
                                    flagged: true})""",
                                  id=sid, name=name, type=stype, doc=doc['id'])
                            shadow_entities[name] = sid
                            self.stats['shadow_entities'] += 1

                            s.run("""MATCH (d:Document {id: $doc}), (s:ShadowEntity {id: $sid})
                                    CREATE (d)-[:MENTIONS {mention_text: $name,
                                    confidence: 0.4, flagged: true}]->(s)""",
                                  doc=doc['id'], sid=sid, name=name)
                            self.stats['shadow_mentions'] += 1

        print(f"{self.stats['shadow_entities']} shadow entities")
        print(f"{self.stats['shadow_mentions']} shadow mentions")

    def export_to_csv(self):
        print("\n=== STEP 8: Exporting to CSV ===")

        nodes_df = pd.DataFrame(self.nodes_data)
        edges_df = pd.DataFrame(self.edges_data)

        nodes_csv = self.graph_dir / "nodes.csv"
        edges_csv = self.graph_dir / "edges.csv"

        nodes_df.to_csv(nodes_csv, index=False)
        edges_df.to_csv(edges_csv, index=False)

        print(f"Exported {len(nodes_df)} nodes to {nodes_csv}")
        print(f"Exported {len(edges_df)} edges to {edges_csv}")

    def generate_statistics(self):
        print("\n=== STEP 9: Generating Statistics ===")

        stats_summary = {
            'ingestion_timestamp': datetime.now().isoformat(),
            'neo4j_uri': self.neo4j_uri,
            'nodes_created': {
                'employees': self.stats['employees'],
                'projects': self.stats['projects'],
                'products': self.stats['products'],
                'policies': self.stats['policies'],
                'regulations': self.stats['regulations'],
                'documents': self.stats['documents'],
                'shadow_entities': self.stats['shadow_entities']
            },
            'relationships_created': {
                'reports_to': self.stats['reports_to'],
                'works_on': self.stats['works_on'],
                'owns': self.stats['owns'],
                'sent': self.stats['sent'],
                'sent_to': self.stats['sent_to'],
                'mentions': self.stats['mentions'],
                'shadow_mentions': self.stats['shadow_mentions'],
                'contradictions': self.stats['contradictions']
            },
            'totals': {
                'total_nodes': sum([self.stats['employees'], self.stats['projects'],
                                   self.stats['products'], self.stats['policies'],
                                   self.stats['regulations'], self.stats['documents'],
                                   self.stats['shadow_entities']]),
                'total_relationships': sum([self.stats['reports_to'], self.stats['works_on'],
                                           self.stats['owns'], self.stats['sent'],
                                           self.stats['sent_to'], self.stats['mentions'],
                                           self.stats['shadow_mentions'], self.stats['contradictions']])
            }
        }

        log_path = self.graph_dir / "ingestion_log.json"
        with open(log_path, 'w') as f:
            json.dump(stats_summary, f, indent=2)

        print(f"Statistics saved to {log_path}")
        return stats_summary

    def validate_graph(self):
        print("\n=== STEP 10: Validating Graph ===")

        with self.driver.session() as s:
            # Check for orphaned nodes
            r = s.run("""MATCH (n) WHERE NOT (n)--() 
                        RETURN labels(n)[0] AS type, count(*) AS count""")
            orphans = [(rec['type'], rec['count']) for rec in r]

            if orphans:
                print(f"Found orphaned nodes:")
                for type, count in orphans:
                    print(f"    {type}: {count}")
            else:
                print("No orphaned nodes")

            # Verify ground truth relationships
            r = s.run("MATCH ()-[r {confidence: 1.0}]->() RETURN count(r) AS count")
            gt_count = r.single()['count']
            print(f"{gt_count} ground truth relationships (confidence=1.0)")

            # Count contradictions
            r = s.run("MATCH ()-[r {flagged: true}]->() RETURN count(r) AS count")
            contra_count = r.single()['count']
            print(f"{contra_count} flagged contradictions")

    def run(self):
        print("=" * 60)
        print("Phase 6: Data Ingestion Pipeline")
        print("=" * 60)


        self.connect_neo4j()
        self.clear_database()
        self.create_indexes()

        self.ingest_entities()
        self.ingest_csv_relationships()
        self.ingest_documents()
        self.ingest_email_metadata()
        self.ingest_document_mentions()
        self.ingest_contradictions()
        self.ingest_shadow_entities()

        self.export_to_csv()
        stats = self.generate_statistics()
        self.validate_graph()

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Total nodes: {stats['totals']['total_nodes']}")
        print(f"Total relationships: {stats['totals']['total_relationships']}")
        print(f"\nNext: Phase 7 - Entity Extraction")
        print("=" * 60)

        return stats


if __name__ == "__main__":
    import yaml

    print("=" * 60)
    print("Phase 6: Neo4j Ingestion")
    print("=" * 60)

    # Load configs
    with open("config/pipeline.yaml") as f:
        pipeline_config = yaml.safe_load(f)

    with open("config/neo4j.yaml") as f:
        neo4j_config = yaml.safe_load(f)

    # Get phase-specific settings
    phase_config = pipeline_config['ingestion']

    print(f"Configuration:")
    print(f"  Neo4j URI: {neo4j_config['uri']}")
    print(f"  Clear database: {phase_config['clear_before_load']}")
    print(f"  Batch size: {phase_config['batch_size']}")
    print()

    # Initialize with config values
    ingestor = DataIngestor(
        neo4j_uri=neo4j_config['uri'],
        neo4j_user=neo4j_config['user'],
        neo4j_password=neo4j_config['password'],
        clear_db=phase_config['clear_before_load'],
        batch_size=phase_config['batch_size']
    )

    ingestor.run()