"""
Phase 4: External Document Download
Downloads or provides instructions for obtaining external regulatory and compliance documents.
"""

import requests
import json
from pathlib import Path
from datetime import datetime
import yaml
import time


class ExternalDocumentDownloader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create output directory
        self.output_dir = Path(self.config['paths']['external_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Document sources
        self.documents = [
            {
                'id': 'ext_001',
                'name': 'GDPR_summary.pdf',
                'description': 'GDPR Regulation Summary',
                'url': 'https://gdpr-info.eu/gdpr.pdf',
                'fallback_url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679',
                'type': 'regulation',
                'auto_download': True,
                'entity_references': ['reg_001']  # GDPR
            },
            {
                'id': 'ext_002',
                'name': 'ISO27001_overview.pdf',
                'description': 'ISO 27001 Information Security Standard Overview',
                'url': 'https://www.iso.org/standard/27001',
                'type': 'standard',
                'auto_download': False,  # ISO docs require purchase
                'manual_instructions': """
ISO 27001 Overview - Manual Download Required:
1. Visit: https://www.iso.org/standard/27001
2. Download the free overview/summary document
3. Save as: data/external/ISO27001_overview.pdf

Alternative: Use a publicly available ISO 27001 guide:
- https://www.itgovernance.co.uk/iso27001 (free guides section)
- Search for "ISO 27001 overview PDF" and download a compliance guide
""",
                'entity_references': ['reg_002']  # ISO 27001
            },
            {
                'id': 'ext_003',
                'name': 'AWS_compliance.pdf',
                'description': 'AWS Compliance and Security Documentation',
                'url': 'https://docs.aws.amazon.com/pdfs/whitepapers/latest/aws-overview/aws-overview.pdf',
                'type': 'vendor_documentation',
                'auto_download': True,
                'entity_references': ['prod_002']  # AWS
            },
            {
                'id': 'ext_004',
                'name': 'Salesforce_security.pdf',
                'description': 'Salesforce Security and Compliance Documentation',
                'url': 'https://www.salesforce.com/content/dam/web/en_us/www/documents/white-papers/security-privacy-and-architecture.pdf',
                'fallback_url': 'https://trust.salesforce.com/en/security/',
                'type': 'vendor_documentation',
                'auto_download': True,
                'entity_references': ['prod_001']  # Salesforce
            }
        ]

    def download_file(self, url: str, output_path: Path, timeout: int = 30) -> bool:
        """Download a file from URL with error handling."""
        try:
            print(f"  Attempting download from: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()

            # Check if response is actually a PDF or document
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and 'application' not in content_type.lower():
                print(f"  ⚠️  Warning: Content-Type is {content_type}, may not be a PDF")

            # Write file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_path.stat().st_size
            print(f"  ✅ Downloaded successfully ({file_size:,} bytes)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Download failed: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            return False

    def create_placeholder_document(self, output_path: Path, doc_info: dict) -> bool:
        """Create a placeholder text file with download instructions."""
        try:
            placeholder_content = f"""
================================================================================
EXTERNAL DOCUMENT PLACEHOLDER
================================================================================

Document: {doc_info['name']}
Description: {doc_info['description']}
Type: {doc_info['type']}
Document ID: {doc_info['id']}

This is a placeholder file. The actual document should be obtained from:
{doc_info['url']}

{doc_info.get('manual_instructions', '')}

================================================================================
For Knowledge Graph Demo Purposes:
================================================================================
This placeholder represents an external regulatory/compliance document that
would be ingested into the knowledge graph. In a production system, this would
be the actual PDF document.

Entity References: {', '.join(doc_info['entity_references'])}

Generated: {datetime.now().isoformat()}
================================================================================
"""

            # Save as .txt instead of .pdf for placeholder
            txt_path = output_path.with_suffix('.txt')
            with open(txt_path, 'w') as f:
                f.write(placeholder_content)

            print(f"  ℹ️  Created placeholder: {txt_path.name}")
            return True

        except Exception as e:
            print(f"  ❌ Failed to create placeholder: {e}")
            return False

    def download_all(self) -> dict:
        """Download all external documents."""
        print("=" * 70)
        print("EXTERNAL DOCUMENT DOWNLOAD")
        print("=" * 70)

        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_documents': len(self.documents),
            'documents': []
        }

        successful_downloads = 0
        placeholder_created = 0
        failed_downloads = 0

        for doc in self.documents:
            print(f"\n📄 Processing: {doc['name']}")
            output_path = self.output_dir / doc['name']

            download_success = False
            is_placeholder = False

            if doc['auto_download']:
                # Try primary URL
                download_success = self.download_file(doc['url'], output_path)

                # Try fallback URL if available
                if not download_success and 'fallback_url' in doc:
                    print(f"  Trying fallback URL...")
                    download_success = self.download_file(doc['fallback_url'], output_path)

                # Create placeholder if download failed
                if not download_success:
                    print(f"  Creating placeholder instead...")
                    is_placeholder = self.create_placeholder_document(output_path, doc)
                    if is_placeholder:
                        placeholder_created += 1
                else:
                    successful_downloads += 1
            else:
                # Manual download required
                print(f"  ⚠️  Manual download required")
                print(doc['manual_instructions'])
                is_placeholder = self.create_placeholder_document(output_path, doc)
                if is_placeholder:
                    placeholder_created += 1

            # Check if file exists
            actual_path = output_path if output_path.exists() else output_path.with_suffix('.txt')
            file_exists = actual_path.exists()

            if not file_exists and not is_placeholder:
                failed_downloads += 1

            # Add to metadata
            metadata['documents'].append({
                'id': doc['id'],
                'filename': doc['name'],
                'actual_filename': actual_path.name if file_exists else None,
                'description': doc['description'],
                'type': doc['type'],
                'source_url': doc['url'],
                'download_status': 'success' if download_success else ('placeholder' if is_placeholder else 'failed'),
                'file_exists': file_exists,
                'file_size': actual_path.stat().st_size if file_exists else 0,
                'entity_references': doc['entity_references'],
                'manual_download_required': not doc['auto_download']
            })

            time.sleep(1)  # Be polite with requests

        # Save metadata
        metadata_path = self.output_dir / 'external_docs_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"Total documents: {len(self.documents)}")
        print(f"✅ Successful downloads: {successful_downloads}")
        print(f"ℹ️  Placeholders created: {placeholder_created}")
        print(f"❌ Failed: {failed_downloads}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Metadata saved to: {metadata_path}")

        # Print manual instructions if needed
        manual_docs = [d for d in self.documents if not d['auto_download']]
        if manual_docs:
            print("\n" + "=" * 70)
            print("MANUAL DOWNLOAD INSTRUCTIONS")
            print("=" * 70)
            for doc in manual_docs:
                print(f"\n📋 {doc['name']}:")
                print(doc['manual_instructions'])

        return metadata

    def validate_downloads(self) -> bool:
        """Validate that all required files exist."""
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)

        all_valid = True
        for doc in self.documents:
            expected_path = self.output_dir / doc['name']
            placeholder_path = expected_path.with_suffix('.txt')

            if expected_path.exists():
                file_size = expected_path.stat().st_size
                print(f"✅ {doc['name']}: {file_size:,} bytes")
            elif placeholder_path.exists():
                print(f"ℹ️  {doc['name']}: Placeholder exists (manual download may be needed)")
            else:
                print(f"❌ {doc['name']}: Missing")
                all_valid = False

        return all_valid


def main():
    downloader = ExternalDocumentDownloader()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    EXTERNAL DOCUMENT DOWNLOADER                      ║
║                                                                      ║
║  This script attempts to download external regulatory and vendor    ║
║  documentation for the Knowledge Graph demo.                        ║
║                                                                      ║
║  Note: Some documents may require manual download due to licensing  ║
║  or access restrictions. Placeholders will be created for demo      ║
║  purposes where automatic download is not possible.                 ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Download documents
    metadata = downloader.download_all()

    # Validate
    downloader.validate_downloads()

    print("\n✅ Phase 4 complete!")
    print("\nNext steps:")
    print("  1. Review placeholders in data/external/")
    print("  2. Manually download any missing documents if needed")
    print("  3. Run Phase 5: python generate_metadata.py")


if __name__ == '__main__':
    main()