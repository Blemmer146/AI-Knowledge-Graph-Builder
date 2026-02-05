"""
Master Orchestrator Script
Runs all phases of the CodeFlow Knowledge Graph demo data generation in sequence.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json


class OrchestrationError(Exception):
    """Custom exception for orchestration errors."""
    pass


class KnowledgeGraphOrchestrator:
    def __init__(self):
        self.phases = [
            {
                'id': 1,
                'name': 'Structured Data Generation',
                'script': 'gen_data_str.py',
                'description': 'Generates employees, projects, products, policies CSVs',
                'outputs': [
                    'data/structured/employees.csv',
                    'data/structured/projects.csv',
                    'data/structured/project_assignments.csv',
                    'data/structured/products.csv',
                    'data/structured/policies.csv',
                    'data/entities.json'
                ],
                'critical': True
            },
            {
                'id': 2,
                'name': 'Semi-Structured Reports',
                'script': 'gen_data_semstr.py',
                'description': 'Generates DOCX project status reports with contradictions',
                'outputs': [
                    'data/semi_structured/project_report_001.docx',
                    'data/reports_metadata.json'
                ],
                'critical': True
            },
            {
                'id': 3,
                'name': 'Unstructured Emails',
                'script': 'gen_data_email.py',
                'description': 'Generates emails based on Enron templates with name replacement',
                'outputs': [
                    'data/unstructured/email_001.txt',
                    'data/emails_metadata.json',
                    'data/name_mapping.json'
                ],
                'critical': True
            },
            {
                'id': 4,
                'name': 'External Documents',
                'script': 'gen_data_external.py',
                'description': 'Downloads external regulatory and vendor documentation',
                'outputs': [
                    'data/external/GDPR_summary.pdf',
                    'data/external/external_docs_metadata.json'
                ],
                'critical': False  # Can proceed with placeholders
            },
            {
                'id': 5,
                'name': 'Master Metadata',
                'script': 'gen_data_metadata.py',
                'description': 'Consolidates all metadata into master metadata.json',
                'outputs': [
                    'data/metadata.json'
                ],
                'critical': True
            }
        ]

        self.execution_log = []
        self.start_time = datetime.now()

    def print_banner(self):
        """Print orchestrator banner."""
        print("\n" + "=" * 80)
        print(" " * 20 + "CODEFLOW KNOWLEDGE GRAPH DATA GENERATOR")
        print(" " * 30 + "Master Orchestrator")
        print("=" * 80)
        print(f"\nStarting execution at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total phases: {len(self.phases)}\n")

    def validate_prerequisites(self) -> bool:
        """Validate that all required scripts exist."""
        print("🔍 Validating prerequisites...")

        missing_scripts = []
        for phase in self.phases:
            script_path = Path(phase['script'])
            if not script_path.exists():
                missing_scripts.append(phase['script'])
                print(f"   ❌ Missing: {phase['script']}")
            else:
                print(f"   ✅ Found: {phase['script']}")

        # Check for config.yaml
        if not Path('config.yaml').exists():
            print(f"   ❌ Missing: config.yaml")
            missing_scripts.append('config.yaml')
        else:
            print(f"   ✅ Found: config.yaml")

        if missing_scripts:
            print(f"\n❌ Missing {len(missing_scripts)} required file(s). Cannot proceed.")
            return False

        print("\n✅ All prerequisites validated!\n")
        return True

    def run_phase(self, phase: dict) -> bool:
        """Run a single phase."""
        phase_num = phase['id']
        print("\n" + "=" * 80)
        print(f"PHASE {phase_num}: {phase['name']}")
        print("=" * 80)
        print(f"Script: {phase['script']}")
        print(f"Description: {phase['description']}")
        print()

        # Run the script
        try:
            result = subprocess.run(
                [sys.executable, phase['script']],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Print output
            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                print(f"\n❌ Phase {phase_num} failed with return code {result.returncode}")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr)

                self.execution_log.append({
                    'phase': phase_num,
                    'name': phase['name'],
                    'status': 'failed',
                    'return_code': result.returncode,
                    'error': result.stderr
                })

                return False

            # Validate outputs
            print(f"\n🔍 Validating outputs for Phase {phase_num}...")
            missing_outputs = []
            for output in phase['outputs']:
                output_path = Path(output)
                if output_path.exists():
                    size = output_path.stat().st_size
                    print(f"   ✅ {output} ({size:,} bytes)")
                else:
                    # For external docs, check for .txt placeholders
                    txt_path = output_path.with_suffix('.txt')
                    if txt_path.exists():
                        print(f"   ℹ️  {output} (placeholder exists)")
                    else:
                        print(f"   ⚠️  {output} (missing)")
                        missing_outputs.append(output)

            if missing_outputs and phase['critical']:
                print(f"\n⚠️  Phase {phase_num} completed but some outputs are missing")
                self.execution_log.append({
                    'phase': phase_num,
                    'name': phase['name'],
                    'status': 'completed_with_warnings',
                    'missing_outputs': missing_outputs
                })
            else:
                print(f"\n✅ Phase {phase_num} completed successfully!")
                self.execution_log.append({
                    'phase': phase_num,
                    'name': phase['name'],
                    'status': 'success'
                })

            return True

        except subprocess.TimeoutExpired:
            print(f"\n❌ Phase {phase_num} timed out after 5 minutes")
            self.execution_log.append({
                'phase': phase_num,
                'name': phase['name'],
                'status': 'timeout'
            })
            return False

        except Exception as e:
            print(f"\n❌ Phase {phase_num} failed with exception: {e}")
            self.execution_log.append({
                'phase': phase_num,
                'name': phase['name'],
                'status': 'error',
                'exception': str(e)
            })
            return False

    def run_all(self) -> bool:
        """Run all phases in sequence."""
        self.print_banner()

        # Validate prerequisites
        if not self.validate_prerequisites():
            return False

        # Run each phase
        for phase in self.phases:
            success = self.run_phase(phase)

            if not success and phase['critical']:
                print(f"\n❌ Critical phase failed. Stopping execution.")
                self.print_summary()
                return False

            if not success and not phase['critical']:
                print(f"\n⚠️  Non-critical phase failed. Continuing...")

        self.print_summary()
        return True

    def print_summary(self):
        """Print execution summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        print("\n" + "=" * 80)
        print(" " * 30 + "EXECUTION SUMMARY")
        print("=" * 80)

        print(f"\nStart time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.1f} seconds")

        print("\nPhase Results:")
        for log_entry in self.execution_log:
            status_emoji = {
                'success': '✅',
                'completed_with_warnings': '⚠️ ',
                'failed': '❌',
                'timeout': '⏱️ ',
                'error': '❌'
            }.get(log_entry['status'], '❓')

            print(f"   {status_emoji} Phase {log_entry['phase']}: {log_entry['name']} - {log_entry['status']}")

        # Count successes
        successful = sum(1 for log in self.execution_log if log['status'] in ['success', 'completed_with_warnings'])
        total = len(self.execution_log)

        print(f"\nOverall: {successful}/{total} phases completed")

        if successful == total:
            print("\n🎉 All phases completed successfully!")
            print("\n📁 Generated Data Structure:")
            print("""
data/
├── structured/
│   ├── employees.csv (30 employees)
│   ├── projects.csv (12 projects)
│   ├── project_assignments.csv (30-40 assignments)
│   ├── products.csv (7 products)
│   └── policies.csv (5 policies)
├── semi_structured/
│   └── project_report_*.docx (6 reports)
├── unstructured/
│   └── email_*.txt (15 emails)
├── external/
│   └── *.pdf (4 external documents)
├── entities.json (canonical entity registry)
├── metadata.json (master metadata)
├── reports_metadata.json
├── emails_metadata.json
└── name_mapping.json
            """)

            print("\n📋 Next Steps:")
            print("   1. Review data/metadata.json for statistics and validation baseline")
            print("   2. Build knowledge graph ingestion pipeline")
            print("   3. Extract entities and relationships from documents")
            print("   4. Compare extracted relationships vs. validation_baseline")
            print("   5. Identify and analyze contradictions")

            print("\n💡 Key Validation Points:")
            print("   - Structured data (CSV) = 100% confidence ground truth")
            print("   - Semi-structured (reports) = ~70% alignment (30% contradictions)")
            print("   - Unstructured (emails) = ~60-70% alignment")
            print("   - Co-occurrence ≠ relationship (track provenance!)")
        else:
            print("\n⚠️  Some phases failed. Review errors above.")

        # Save execution log
        log_path = Path('data/execution_log.json')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'phases': self.execution_log
            }, f, indent=2)

        print(f"\n📝 Execution log saved to: {log_path}")
        print("=" * 80 + "\n")


def main():
    orchestrator = KnowledgeGraphOrchestrator()
    success = orchestrator.run_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()