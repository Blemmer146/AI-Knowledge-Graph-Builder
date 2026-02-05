"""
Pipeline Orchestrator
Executes knowledge graph pipeline phases in sequence

Usage:
    python run_pipeline.py                 # Run all phases
    python run_pipeline.py --phase 7       # Run single phase
    python run_pipeline.py --from 7        # Run from phase 7 onward
    python run_pipeline.py --validate      # Run validation only
"""

import yaml
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess


class PipelineOrchestrator:
    """Orchestrates the execution of pipeline phases"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.project_root = Path.cwd()

        # Load configurations
        self.pipeline_config = self._load_yaml(self.config_dir / "pipeline.yaml")
        self.neo4j_config = self._load_yaml(self.config_dir / "neo4j.yaml")
        self.ollama_config = self._load_yaml(self.config_dir / "ollama.yaml")

        # Phase definitions
        self.phases = {
            6: {
                'name': 'Neo4j Ingestion',
                'script': 'neo4j_loader.py',
                'config_key': 'ingestion',
                'description': 'Load data into Neo4j graph database'
            },
            7: {
                'name': 'Entity Extraction',
                'script': 'entity_extractor.py',
                'config_key': 'extraction',
                'description': 'Extract entities and relationships from documents'
            },
            8: {
                'name': 'Embeddings & Indexing',
                'script': 'embedding_generator.py',
                'config_key': 'embedding',
                'description': 'Generate embeddings and build FAISS indexes'
            },
            9: {
                'name': 'RAG Pipeline',
                'script': 'rag_system.py',
                'config_key': 'rag',
                'description': 'Initialize RAG query system'
            }
        }

        # Execution state
        self.execution_log = []
        self.start_time = None
        self.end_time = None

    def _load_yaml(self, path: Path) -> dict:
        """Load YAML configuration file"""
        if not path.exists():
            print(f"ERROR: Configuration file not found: {path}")
            sys.exit(1)

        with open(path) as f:
            return yaml.safe_load(f)

    def _check_phase_enabled(self, phase_num: int) -> bool:
        """Check if phase is enabled in config"""
        config_key = self.phases[phase_num]['config_key']
        return self.pipeline_config.get(config_key, {}).get('enabled', True)

    def _check_prerequisites(self, phase_num: int) -> bool:
        """Check if phase prerequisites are met"""
        if phase_num == 6:
            # Check if source data exists
            source_dir = Path(self.pipeline_config['paths']['source'])
            if not source_dir.exists():
                print(f"ERROR: Source data directory not found: {source_dir}")
                return False

        elif phase_num == 7:
            # Check if Neo4j ingestion completed
            graph_log = Path(self.pipeline_config['ingestion']['outputs']['log'])
            if not graph_log.exists():
                print(f"ERROR: Phase 6 must complete first (missing {graph_log})")
                return False

        elif phase_num == 8:
            # Check if extraction completed
            triples = Path(self.pipeline_config['extraction']['outputs']['triples'])
            if not triples.exists():
                print(f"ERROR: Phase 7 must complete first (missing {triples})")
                return False

        elif phase_num == 9:
            # Check if embeddings exist
            triple_index = Path(self.pipeline_config['embedding']['outputs']['triple_index'])
            if not triple_index.exists():
                print(f"ERROR: Phase 8 must complete first (missing {triple_index})")
                return False

        return True

    def run_phase(self, phase_num: int) -> bool:
        """Execute a single phase"""
        if phase_num not in self.phases:
            print(f"ERROR: Invalid phase number: {phase_num}")
            return False

        phase = self.phases[phase_num]

        print("\n" + "=" * 70)
        print(f"PHASE {phase_num}: {phase['name']}")
        print("=" * 70)
        print(f"Description: {phase['description']}")
        print(f"Script: {phase['script']}")

        # Check if enabled
        if not self._check_phase_enabled(phase_num):
            print(f"⊘ Phase {phase_num} is disabled in config")
            self._log_phase(phase_num, 'skipped', 'Disabled in config')
            return True

        # Check prerequisites
        if not self._check_prerequisites(phase_num):
            print(f"✗ Prerequisites not met for Phase {phase_num}")
            self._log_phase(phase_num, 'failed', 'Prerequisites not met')
            return False

        # Check if script exists
        script_path = self.project_root / phase['script']
        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            self._log_phase(phase_num, 'failed', 'Script not found')
            return False

        # Execute phase
        print(f"\n▶ Starting execution...")
        phase_start = datetime.now()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )

            phase_end = datetime.now()
            duration = (phase_end - phase_start).total_seconds()

            # Print output
            if result.stdout:
                print(result.stdout)

            if result.returncode == 0:
                print(f"✓ Phase {phase_num} completed successfully ({duration:.1f}s)")
                self._log_phase(phase_num, 'success', '', duration)
                return True
            else:
                print(f"✗ Phase {phase_num} failed")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr)
                self._log_phase(phase_num, 'failed', result.stderr, duration)
                return False

        except Exception as e:
            phase_end = datetime.now()
            duration = (phase_end - phase_start).total_seconds()
            print(f"✗ Exception during Phase {phase_num}: {e}")
            self._log_phase(phase_num, 'failed', str(e), duration)
            return False

    def run_all(self, start_from: Optional[int] = None):
        """Run all phases in sequence"""
        print("\n" + "=" * 70)
        print("KNOWLEDGE GRAPH PIPELINE ORCHESTRATOR")
        print("=" * 70)
        print(f"Project: {self.pipeline_config['project']['name']}")
        print(f"Version: {self.pipeline_config['project']['version']}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.start_time = datetime.now()

        # Determine which phases to run
        phases_to_run = [p for p in sorted(self.phases.keys())
                         if start_from is None or p >= start_from]

        print(f"\nPhases to execute: {', '.join(map(str, phases_to_run))}")
        print("=" * 70)

        # Execute phases
        success_count = 0
        failed_count = 0
        skipped_count = 0

        for phase_num in phases_to_run:
            success = self.run_phase(phase_num)

            if success:
                # Check if it was actually skipped
                if self.execution_log[-1]['status'] == 'skipped':
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                failed_count += 1

                # Ask if we should continue
                if phase_num < max(phases_to_run):
                    response = input(f"\nPhase {phase_num} failed. Continue? (y/n): ")
                    if response.lower() != 'y':
                        print("Pipeline execution stopped by user")
                        break

        self.end_time = datetime.now()

        # Print summary
        self._print_summary(success_count, failed_count, skipped_count)

        # Save execution log
        self._save_execution_log()

    def _log_phase(self, phase_num: int, status: str, error: str = '', duration: float = 0.0):
        """Log phase execution"""
        self.execution_log.append({
            'phase': phase_num,
            'name': self.phases[phase_num]['name'],
            'status': status,
            'error': error,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        })

    def _print_summary(self, success: int, failed: int, skipped: int):
        """Print execution summary"""
        total_duration = (self.end_time - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total duration: {total_duration:.1f}s ({total_duration / 60:.1f}m)")
        print(f"Successful: {success}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print()

        # Print phase details
        for log in self.execution_log:
            status_icon = {
                'success': '✓',
                'failed': '✗',
                'skipped': '⊘'
            }.get(log['status'], '?')

            print(f"  {status_icon} Phase {log['phase']}: {log['name']} "
                  f"({log['duration_seconds']:.1f}s) - {log['status'].upper()}")

        print("=" * 70)

        if failed == 0:
            print("🎉 All phases completed successfully!")
        else:
            print("⚠️  Some phases failed - check logs for details")

    def _save_execution_log(self):
        """Save execution log to file"""
        log_dir = Path(self.pipeline_config['paths']['logs']) / "pipeline_runs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"run_{timestamp}.json"

        log_data = {
            'project': self.pipeline_config['project']['name'],
            'version': self.pipeline_config['project']['version'],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'phases': self.execution_log
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Also save as latest.json
        latest_log = log_dir / "latest.json"
        with open(latest_log, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n📝 Execution log saved: {log_file}")

    def get_status(self):
        """Check status of all phases"""
        print("\n" + "=" * 70)
        print("PIPELINE STATUS")
        print("=" * 70)

        for phase_num in sorted(self.phases.keys()):
            phase = self.phases[phase_num]
            config_key = phase['config_key']

            # Check if enabled
            enabled = self.pipeline_config.get(config_key, {}).get('enabled', True)

            # Check if outputs exist
            outputs_config = self.pipeline_config.get(config_key, {}).get('outputs', {})
            outputs_exist = True

            if isinstance(outputs_config, dict):
                for key, path in outputs_config.items():
                    if isinstance(path, str) and not path.endswith('*'):
                        if not Path(path).exists():
                            outputs_exist = False
                            break

            # Status indicator
            if not enabled:
                status = "⊘ DISABLED"
            elif outputs_exist:
                status = "✓ COMPLETE"
            else:
                status = "○ NOT RUN"

            print(f"{status} - Phase {phase_num}: {phase['name']}")

        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Knowledge Graph Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                 # Run all phases
  python run_pipeline.py --phase 7       # Run phase 7 only
  python run_pipeline.py --from 8        # Run phases 8-9
  python run_pipeline.py --status        # Check pipeline status
        """
    )

    parser.add_argument('--phase', type=int, choices=[6, 7, 8, 9],
                        help='Run specific phase only')
    parser.add_argument('--from', dest='from_phase', type=int, choices=[6, 7, 8, 9],
                        help='Run from specified phase onward')
    parser.add_argument('--status', action='store_true',
                        help='Check status of all phases')
    parser.add_argument('--config-dir', default='config',
                        help='Configuration directory (default: config)')

    args = parser.parse_args()

    # Initialize orchestrator
    try:
        orchestrator = PipelineOrchestrator(config_dir=args.config_dir)
    except Exception as e:
        print(f"ERROR: Failed to initialize orchestrator: {e}")
        sys.exit(1)

    # Execute requested operation
    if args.status:
        orchestrator.get_status()
    elif args.phase:
        success = orchestrator.run_phase(args.phase)
        sys.exit(0 if success else 1)
    elif args.from_phase:
        orchestrator.run_all(start_from=args.from_phase)
    else:
        orchestrator.run_all()


if __name__ == "__main__":
    main()