"""
Phase 5: Metadata Generation
Consolidates metadata from all phases into a master metadata.json file.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import yaml
import csv


class MetadataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = Path(self.config['paths']['output_dir'])

        # Load all metadata files
        self.entities = self._load_json('entities.json')
        self.reports_metadata = self._load_json('reports_metadata.json')
        self.emails_metadata = self._load_json('emails_metadata.json')
        self.external_metadata = self._load_json('external/external_docs_metadata.json')
        self.name_mapping = self._load_json('name_mapping.json')

        # Load CSV ground truth
        self.structured_data = self._load_structured_data()

    def _load_json(self, relative_path: str) -> dict:
        """Load JSON file with error handling."""
        path = self.output_dir / relative_path
        if not path.exists():
            print(f"⚠️  Warning: {path} not found, using empty dict")
            return {}

        with open(path, 'r') as f:
            return json.load(f)

    def _load_structured_data(self) -> Dict:
        """Load all CSV files as ground truth."""
        structured_dir = Path(self.config['paths']['structured_dir'])

        data = {
            'employees': [],
            'projects': [],
            'project_assignments': [],
            'products': [],
            'policies': []
        }

        # Load employees
        emp_path = structured_dir / 'employees.csv'
        if emp_path.exists():
            with open(emp_path, 'r') as f:
                reader = csv.DictReader(f)
                data['employees'] = list(reader)

        # Load projects
        proj_path = structured_dir / 'projects.csv'
        if proj_path.exists():
            with open(proj_path, 'r') as f:
                reader = csv.DictReader(f)
                data['projects'] = list(reader)

        # Load project assignments
        assign_path = structured_dir / 'project_assignments.csv'
        if assign_path.exists():
            with open(assign_path, 'r') as f:
                reader = csv.DictReader(f)
                data['project_assignments'] = list(reader)

        # Load products
        prod_path = structured_dir / 'products.csv'
        if prod_path.exists():
            with open(prod_path, 'r') as f:
                reader = csv.DictReader(f)
                data['products'] = list(reader)

        # Load policies
        pol_path = structured_dir / 'policies.csv'
        if pol_path.exists():
            with open(pol_path, 'r') as f:
                reader = csv.DictReader(f)
                data['policies'] = list(reader)

        return data

    def _build_validation_baseline(self) -> Dict:
        """Build ground truth validation baseline from structured data."""
        baseline = {
            'description': 'Ground truth from structured data (CSV files)',
            'confidence': 1.0,
            'source': 'structured_data',
            'employee_project_assignments': [],
            'employee_details': [],
            'project_details': [],
            'product_catalog': [],
            'policy_catalog': []
        }

        # Employee-project assignments (ground truth)
        for assignment in self.structured_data['project_assignments']:
            baseline['employee_project_assignments'].append({
                'employee_id': assignment['employee_id'],
                'project_id': assignment['project_id'],
                'role': assignment.get('role', 'Unknown'),
                'source': 'project_assignments.csv'
            })

        # Employee details
        for emp in self.structured_data['employees']:
            baseline['employee_details'].append({
                'employee_id': emp['employee_id'],
                'full_name': emp['full_name'],
                'email': emp['email'],
                'department': emp['department'],
                'role': emp['role'],
                'manager_id': emp.get('manager_id', ''),
                'source': 'employees.csv'
            })

        # Project details
        for proj in self.structured_data['projects']:
            baseline['project_details'].append({
                'project_id': proj['project_id'],
                'name': proj['name'],
                'status': proj['status'],
                'department': proj['department'],
                'owner_id': proj.get('owner_id', ''),
                'source': 'projects.csv'
            })

        # Product catalog
        for prod in self.structured_data['products']:
            baseline['product_catalog'].append({
                'product_id': prod['product_id'],
                'name': prod['name'],
                'vendor': prod['vendor'],
                'category': prod['category'],
                'source': 'products.csv'
            })

        # Policy catalog
        for pol in self.structured_data['policies']:
            baseline['policy_catalog'].append({
                'policy_id': pol['policy_id'],
                'name': pol['name'],
                'category': pol['category'],
                'version': pol['version'],
                'source': 'policies.csv'
            })

        return baseline

    def _calculate_statistics(self) -> Dict:
        """Calculate statistics across all data sources."""
        stats = {
            'entity_counts': {
                'employees': len(self.entities.get('employees', [])),
                'projects': len(self.entities.get('projects', [])),
                'products': len(self.entities.get('products', [])),
                'policies': len(self.entities.get('policies', [])),
                'regulations': len(self.entities.get('regulations', []))
            },
            'document_counts': {
                'structured_csvs': 5,  # employees, projects, assignments, products, policies
                'semi_structured_reports': self.reports_metadata.get('total_reports', 0),
                'unstructured_emails': self.emails_metadata.get('total_emails', 0),
                'external_documents': self.external_metadata.get('total_documents', 0)
            },
            'contradiction_statistics': {
                'reports_with_contradictions': 0,
                'emails_with_contradictions': 0,
                'total_contradictions': 0,
                'contradiction_types': {}
            },
            'entity_mention_statistics': {
                'total_entity_mentions': 0,
                'mentions_by_type': {},
                'documents_with_mentions': 0
            }
        }

        # Count contradictions in reports
        for doc in self.reports_metadata.get('documents', []):
            contradictions = doc.get('contradictions', [])
            if contradictions:
                stats['contradiction_statistics']['reports_with_contradictions'] += 1
                stats['contradiction_statistics']['total_contradictions'] += len(contradictions)

                for contradiction in contradictions:
                    ctype = contradiction['type']
                    stats['contradiction_statistics']['contradiction_types'][ctype] = \
                        stats['contradiction_statistics']['contradiction_types'].get(ctype, 0) + 1

        # Count contradictions in emails
        for doc in self.emails_metadata.get('documents', []):
            contradictions = doc.get('contradictions', [])
            if contradictions:
                stats['contradiction_statistics']['emails_with_contradictions'] += 1
                stats['contradiction_statistics']['total_contradictions'] += len(contradictions)

                for contradiction in contradictions:
                    ctype = contradiction['type']
                    stats['contradiction_statistics']['contradiction_types'][ctype] = \
                        stats['contradiction_statistics']['contradiction_types'].get(ctype, 0) + 1

        # Count entity mentions
        all_docs = (self.reports_metadata.get('documents', []) +
                    self.emails_metadata.get('documents', []))

        for doc in all_docs:
            mentions = doc.get('entities_mentioned', [])
            if mentions:
                stats['entity_mention_statistics']['documents_with_mentions'] += 1
                stats['entity_mention_statistics']['total_entity_mentions'] += len(mentions)

                for mention in mentions:
                    mtype = mention.get('entity_type', 'unknown')
                    stats['entity_mention_statistics']['mentions_by_type'][mtype] = \
                        stats['entity_mention_statistics']['mentions_by_type'].get(mtype, 0) + 1

        return stats

    def _merge_documents(self) -> List[Dict]:
        """Merge all document metadata."""
        documents = []

        # Add reports
        for doc in self.reports_metadata.get('documents', []):
            documents.append({
                **doc,
                'data_source': 'semi_structured_reports',
                'file_path': f"data/semi_structured/{doc['filename']}"
            })

        # Add emails
        for doc in self.emails_metadata.get('documents', []):
            documents.append({
                **doc,
                'data_source': 'unstructured_emails',
                'file_path': f"data/unstructured/{doc['filename']}"
            })

        # Add external docs
        for doc in self.external_metadata.get('documents', []):
            documents.append({
                **doc,
                'data_source': 'external_documents',
                'file_path': f"data/external/{doc.get('actual_filename', doc['filename'])}"
            })

        return documents

    def generate_master_metadata(self) -> Dict:
        """Generate consolidated master metadata."""
        print("=" * 70)
        print("GENERATING MASTER METADATA")
        print("=" * 70)

        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'project_info': {
                'name': 'CodeFlow Knowledge Graph Demo',
                'description': 'Enterprise knowledge graph demonstration with intentional contradictions',
                'company': self.config['company']['name'],
                'domain': self.config['company']['domain'],
                'random_seed': self.config['generation']['random_seed']
            },
            'statistics': self._calculate_statistics(),
            'validation_baseline': self._build_validation_baseline(),
            'documents': self._merge_documents(),
            'entity_registry': self.entities,
            'name_mapping': {
                'description': 'Mapping from Enron email templates to CodeFlow entities',
                'mappings': self.name_mapping
            },
            'data_quality_notes': {
                'structured_data_confidence': 1.0,
                'semi_structured_alignment': '~70% (30% intentional contradictions)',
                'unstructured_alignment': '~60-70% (20-30% intentional contradictions)',
                'contradiction_strategy': 'Intentional contradictions for knowledge graph validation',
                'mention_vs_relationship': 'Co-occurrence in documents ≠ semantic relationship'
            },
            'phases_completed': {
                'phase_1_structured': True,
                'phase_2_semi_structured': True,
                'phase_3_unstructured': True,
                'phase_4_external': True,
                'phase_5_metadata': True
            }
        }

        # Save master metadata
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Master metadata generated: {metadata_path}")

        # Print summary
        self._print_summary(metadata)

        return metadata

    def _print_summary(self, metadata: Dict):
        """Print metadata summary."""
        print("\n" + "=" * 70)
        print("METADATA SUMMARY")
        print("=" * 70)

        stats = metadata['statistics']

        print("\n📊 Entity Counts:")
        for entity_type, count in stats['entity_counts'].items():
            print(f"   {entity_type}: {count}")

        print("\n📄 Document Counts:")
        for doc_type, count in stats['document_counts'].items():
            print(f"   {doc_type}: {count}")

        print("\n⚠️  Contradiction Statistics:")
        cs = stats['contradiction_statistics']
        print(f"   Reports with contradictions: {cs['reports_with_contradictions']}")
        print(f"   Emails with contradictions: {cs['emails_with_contradictions']}")
        print(f"   Total contradictions: {cs['total_contradictions']}")
        print(f"   Contradiction types:")
        for ctype, count in cs['contradiction_types'].items():
            print(f"      {ctype}: {count}")

        print("\n🔗 Entity Mention Statistics:")
        ms = stats['entity_mention_statistics']
        print(f"   Total entity mentions: {ms['total_entity_mentions']}")
        print(f"   Documents with mentions: {ms['documents_with_mentions']}")
        print(f"   Mentions by type:")
        for mtype, count in ms['mentions_by_type'].items():
            print(f"      {mtype}: {count}")

        print("\n✅ Validation Baseline:")
        vb = metadata['validation_baseline']
        print(f"   Employee-project assignments: {len(vb['employee_project_assignments'])}")
        print(f"   Employee records: {len(vb['employee_details'])}")
        print(f"   Project records: {len(vb['project_details'])}")
        print(f"   Products: {len(vb['product_catalog'])}")
        print(f"   Policies: {len(vb['policy_catalog'])}")

        print("\n" + "=" * 70)
        print("All phases completed successfully! 🎉")
        print("=" * 70)


def main():
    generator = MetadataGenerator()
    metadata = generator.generate_master_metadata()

    print("\n📋 Next Steps:")
    print("   1. Review metadata.json for completeness")
    print("   2. Use validation_baseline for knowledge graph validation")
    print("   3. Build knowledge graph ingestion pipeline")
    print("   4. Compare extracted relationships vs. ground truth")
    print("\n💡 Key Files:")
    print("   - data/metadata.json (master metadata)")
    print("   - data/entities.json (entity registry)")
    print("   - data/structured/* (ground truth CSV files)")
    print("   - data/semi_structured/* (project reports with contradictions)")
    print("   - data/unstructured/* (emails with entity mentions)")
    print("   - data/external/* (regulatory/vendor documents)")


if __name__ == '__main__':
    main()