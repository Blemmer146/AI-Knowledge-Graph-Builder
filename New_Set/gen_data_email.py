"""
Phase 3: Unstructured Email Generation
Generates realistic emails based on Enron-style templates with deterministic name replacement.
"""

import json
import random
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
import yaml

# Enron email templates (sanitized samples)
ENRON_EMAIL_TEMPLATES = [
    {
        "from": "ken.lay@enron.com",
        "to": "jeff.skilling@enron.com",
        "subject": "Broadband deployment update",
        "body": """Jeff,

Wanted to give you a quick update on the broadband deployment. We're making good progress, but I'm concerned about the Oracle integration timeline. The team estimates we'll need another 3 weeks.

Can we sync up tomorrow to discuss mitigation strategies?

Ken"""
    },
    {
        "from": "sherron.watkins@enron.com",
        "to": "ken.lay@enron.com",
        "subject": "Accounting concerns - Project Raptor",
        "body": """Ken,

I need to bring to your attention some accounting irregularities I've discovered in Project Raptor. The numbers don't reconcile with our Data Retention Policy, and I believe we may have compliance issues.

This is time-sensitive. Can we meet this week?

Sherron"""
    },
    {
        "from": "jeff.skilling@enron.com",
        "to": "kenneth.rice@enron.com",
        "subject": "Re: Q4 trading platform",
        "body": """Kenneth,

The trading platform migration is behind schedule. I spoke with the Oracle team and they're confident they can deliver by end of month, but I'm skeptical.

What's your assessment? Should we consider alternatives?

Also, have you reviewed the latest version of our Risk Management Policy?

Jeff"""
    },
    {
        "from": "louise.kitchen@enron.com",
        "to": "john.arnold@enron.com",
        "subject": "EnronOnline performance metrics",
        "body": """John,

Great work on EnronOnline last quarter! The performance numbers are impressive. I wanted to discuss expanding the team for the next phase.

Are you available Tuesday afternoon? We should also loop in Greg from infrastructure.

Louise"""
    },
    {
        "from": "vince.kaminski@enron.com",
        "to": "sherron.watkins@enron.com",
        "subject": "Risk model validation",
        "body": """Sherron,

I've completed the risk model validation you requested. The models look sound, but I have concerns about the underlying assumptions in Project Fastow.

Can we schedule 30 minutes to walk through the findings? I'd like to get your perspective before presenting to leadership.

Vince"""
    },
    {
        "from": "greg.whalley@enron.com",
        "to": "louise.kitchen@enron.com",
        "subject": "Infrastructure upgrade - urgent",
        "body": """Louise,

We need to expedite the mainframe upgrade. Current capacity won't support Q1 trading volumes. I've reached out to IBM but their timeline is 8 weeks.

Thoughts on alternatives? Maybe we should look at distributed systems instead.

Greg"""
    },
    {
        "from": "john.arnold@enron.com",
        "to": "vince.kaminski@enron.com",
        "subject": "Trading algorithm review",
        "body": """Vince,

Can you review the new trading algorithm before we deploy? I want to make sure it complies with our Trading Guidelines and doesn't introduce unnecessary risk.

The team is using some new analytics software - I think it's called SAS or something similar. Not sure if it's approved.

John"""
    },
    {
        "from": "kenneth.rice@enron.com",
        "to": "jeff.skilling@enron.com",
        "subject": "Broadband division update",
        "body": """Jeff,

Quick update on broadband division:
- Customer acquisition up 15%
- Network infrastructure on track
- Project Braveheart entering final testing

One concern: we're using some collaboration tools that might not be in our approved software list. The team finds them more efficient than our standard tools.

Kenneth"""
    },
    {
        "from": "sherron.watkins@enron.com",
        "to": "vince.kaminski@enron.com",
        "subject": "Re: Risk model validation",
        "body": """Vince,

Thanks for the thorough analysis. Your concerns about Project Fastow are valid. I think we need to escalate this to Ken and possibly the board.

Before we do that, can you document the specific policy violations you've identified? We'll need that for the audit trail.

Let's meet Friday morning.

Sherron"""
    },
    {
        "from": "louise.kitchen@enron.com",
        "to": "greg.whalley@enron.com",
        "subject": "Re: Infrastructure upgrade - urgent",
        "body": """Greg,

Agree on the urgency. I've been pushing for cloud infrastructure for months - the mainframe approach is outdated.

I had a good conversation with someone from AWS last week. They think they can get us up and running in 4 weeks. Much faster than IBM's timeline.

Want to set up a call with them?

Louise"""
    },
    {
        "from": "jeff.skilling@enron.com",
        "to": "ken.lay@enron.com",
        "subject": "Board presentation next week",
        "body": """Ken,

Preparing for next week's board presentation. I'll cover:
1. Q4 financial results
2. Project Phoenix status
3. Risk management updates
4. Technology roadmap

Do you want me to address the accounting questions that have been raised? Or would you prefer to handle that separately?

Jeff"""
    },
    {
        "from": "greg.whalley@enron.com",
        "to": "john.arnold@enron.com",
        "subject": "Trading desk expansion",
        "body": """John,

Love the idea of expanding the trading desk. Before we proceed, I need you to work with HR on the hiring plan.

Also, make sure all new hires complete the Ethics Training and Information Security Policy certification within their first week.

Greg"""
    },
    {
        "from": "vince.kaminski@enron.com",
        "to": "louise.kitchen@enron.com",
        "subject": "Model deployment timeline",
        "body": """Louise,

The new risk models are ready for deployment. I've tested them against historical data and they perform well.

One issue: the models require data from our CRM system, and I'm not sure our current Oracle setup can handle the query volume. Might need some database optimization.

Can your team help with that?

Vince"""
    },
    {
        "from": "kenneth.rice@enron.com",
        "to": "sherron.watkins@enron.com",
        "subject": "Compliance review request",
        "body": """Sherron,

I need your help reviewing the compliance status of Project Braveheart. We're planning to launch next month but I want to make sure we're not missing anything.

Specifically, I'm concerned about:
- Data privacy requirements
- Financial reporting standards
- Customer contract terms

Can you do a review this week?

Kenneth"""
    },
    {
        "from": "john.arnold@enron.com",
        "to": "greg.whalley@enron.com",
        "subject": "Q1 planning",
        "body": """Greg,

Starting Q1 planning for the trading desk. Based on current projections, I think we need to increase our technology budget by 20%.

Main drivers:
- New trading platforms
- Enhanced analytics (thinking about subscribing to Bloomberg terminal)
- Infrastructure upgrades

Thoughts?

John"""
    }
]

# Shadow IT products (NOT in products.csv)
SHADOW_IT = [
    "Microsoft Teams", "Zoom", "Notion", "Dropbox", "Google Workspace",
    "Monday.com", "Asana", "Trello", "Confluence", "Monday", "Bloomberg Terminal"
]

# Non-existent policies
FAKE_POLICIES = [
    "Remote Work Policy", "AI Usage Guidelines", "Social Media Policy",
    "BYOD Policy", "Travel Expense Policy", "Ethics Training",
    "Trading Guidelines", "Customer Data Policy"
]

# External people/entities
EXTERNAL_ENTITIES = [
    "Jane from VendorCorp", "the AWS team", "someone from Salesforce",
    "our contact at the vendor", "the external consultant",
    "Mike from the audit firm", "the compliance advisor"
]


class EmailGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_seed = self.config['generation']['random_seed']
        random.seed(self.random_seed)

        # Load entities
        entities_path = Path(self.config['paths']['output_dir']) / 'entities.json'
        with open(entities_path, 'r') as f:
            self.entities = json.load(f)

        # Create output directory
        self.output_dir = Path(self.config['paths']['unstructured_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create name mapping
        self.name_mapping = self._create_name_mapping()

        # Products for contextual replacement
        self.products = {p['name'] for p in self.entities['products']}

    def _create_name_mapping(self) -> Dict[str, Dict[str, str]]:
        """Create deterministic mapping from Enron names to CodeFlow names."""
        # Extract unique Enron names from templates
        enron_names = set()
        for template in ENRON_EMAIL_TEMPLATES:
            # Extract from email addresses
            from_name = template['from'].split('@')[0].replace('.', ' ').title()
            to_name = template['to'].split('@')[0].replace('.', ' ').title()
            enron_names.add(from_name)
            enron_names.add(to_name)

            # Extract from body (simple patterns)
            body_names = re.findall(r'\b([A-Z][a-z]+)\b(?!\s+[a-z])', template['body'])
            enron_names.update(body_names)

        enron_names = sorted(list(enron_names))  # Deterministic order

        # Create mapping to CodeFlow employees
        mapping = {}
        employees = self.entities['employees'][:len(enron_names)]

        for enron_name, employee in zip(enron_names, employees):
            mapping[enron_name] = {
                'full_name': employee['full_name'],
                'first_name': employee['first_name'],
                'last_name': employee['last_name'],
                'email': employee['email']
            }

        return mapping

    def _replace_names(self, text: str) -> str:
        """Replace Enron names with CodeFlow names."""
        result = text

        # Replace email addresses first
        for enron_name, codeflow in self.name_mapping.items():
            enron_email_pattern = enron_name.lower().replace(' ', '.')
            result = re.sub(
                rf'\b{enron_email_pattern}@enron\.com\b',
                codeflow['email'],
                result,
                flags=re.IGNORECASE
            )

        # Replace full names and first names
        for enron_name, codeflow in self.name_mapping.items():
            # Replace full name
            result = re.sub(
                rf'\b{re.escape(enron_name)}\b',
                codeflow['full_name'],
                result
            )
            # Replace first name only (when used alone)
            first_name = enron_name.split()[0]
            if random.random() < 0.3:  # 30% chance to use first name only
                result = re.sub(
                    rf'\b{first_name}\b(?!\s+\w)',
                    codeflow['first_name'],
                    result
                )

        return result

    def _replace_company_products(self, text: str) -> str:
        """Replace Enron with CodeFlow and products contextually."""
        result = text.replace('Enron', 'CodeFlow')
        result = result.replace('enron', 'codeflow')

        # Replace products contextually
        product_replacements = {
            'Oracle': 'Salesforce',
            'mainframe': 'AWS',
            'IBM': 'AWS',
            'SAS': 'Tableau',
            'CRM system': 'Salesforce'
        }

        for old_prod, new_prod in product_replacements.items():
            result = re.sub(
                rf'\b{re.escape(old_prod)}\b',
                new_prod,
                result,
                flags=re.IGNORECASE
            )

        return result

    def _inject_contradictions(self, text: str, email_id: int) -> Tuple[str, List[Dict]]:
        """Inject contradictions: shadow IT, fake policies, external entities."""
        contradictions = []
        result = text

        # Every 5th email (~20%) gets contradictions
        if email_id % 5 == 0:
            # Inject shadow IT
            if random.random() < 0.6:
                shadow_product = random.choice(SHADOW_IT)
                insertion_point = result.find('\n\n')
                if insertion_point > 0:
                    insert_text = f"\n\nBTW, the team is using {shadow_product} for collaboration - much better than our standard tools.\n"
                    result = result[:insertion_point] + insert_text + result[insertion_point:]
                    contradictions.append({
                        'type': 'shadow_it',
                        'product': shadow_product,
                        'explanation': f'Mentions {shadow_product} which is not in approved products list'
                    })

            # Inject fake policy
            if random.random() < 0.4:
                fake_policy = random.choice(FAKE_POLICIES)
                result = result.replace(
                    'policy',
                    f'{fake_policy}',
                    1
                )
                contradictions.append({
                    'type': 'fake_policy',
                    'policy': fake_policy,
                    'explanation': f'References {fake_policy} which does not exist in policies.csv'
                })

        # 30% chance to mention external entity
        if random.random() < 0.3:
            external = random.choice(EXTERNAL_ENTITIES)
            sentences = result.split('.')
            if len(sentences) > 2:
                sentences[1] += f" I also spoke with {external}"
                result = '.'.join(sentences)
                contradictions.append({
                    'type': 'external_entity',
                    'entity': external,
                    'explanation': f'Mentions {external} who is not in employees.csv'
                })

        return result, contradictions

    def _extract_entity_mentions(self, text: str) -> List[Dict]:
        """Extract mentions of entities from entities.json."""
        mentions = []

        # Check for employee mentions
        for emp in self.entities['employees']:
            patterns = [
                emp['full_name'],
                emp['first_name'],
                emp['email']
            ]
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    mentions.append({
                        'entity_type': 'employee',
                        'entity_id': emp['id'],
                        'mention_text': pattern
                    })
                    break

        # Check for project mentions
        for proj in self.entities['projects']:
            if proj['name'].lower() in text.lower():
                mentions.append({
                    'entity_type': 'project',
                    'entity_id': proj['id'],
                    'mention_text': proj['name']
                })

        # Check for product mentions
        for prod in self.entities['products']:
            if prod['name'].lower() in text.lower():
                mentions.append({
                    'entity_type': 'product',
                    'entity_id': prod['id'],
                    'mention_text': prod['name']
                })

        # Check for policy mentions
        for pol in self.entities['policies']:
            if pol['name'].lower() in text.lower():
                mentions.append({
                    'entity_type': 'policy',
                    'entity_id': pol['id'],
                    'mention_text': pol['name']
                })

        return mentions

    def generate_emails(self, num_emails: int = 15) -> Dict:
        """Generate emails based on templates."""
        print(f"Generating {num_emails} emails...")

        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_emails': num_emails,
            'random_seed': self.random_seed,
            'documents': []
        }

        # Generate dates
        start_date = datetime(2024, 1, 1)

        for i in range(num_emails):
            template = ENRON_EMAIL_TEMPLATES[i % len(ENRON_EMAIL_TEMPLATES)]
            email_id = f"email_{i + 1:03d}"

            # Replace names and company/products
            email_body = self._replace_names(template['body'])
            email_body = self._replace_company_products(email_body)
            subject = self._replace_names(template['subject'])
            subject = self._replace_company_products(subject)

            # Inject contradictions
            email_body, contradictions = self._inject_contradictions(email_body, i)

            # Generate date
            email_date = start_date + timedelta(days=i * 15)

            # Extract from/to
            from_email = self._replace_names(template['from']).replace('@enron.com', '@codeflow.com')
            to_email = self._replace_names(template['to']).replace('@enron.com', '@codeflow.com')

            # Create email text
            email_text = f"""From: {from_email}
To: {to_email}
Subject: {subject}
Date: {email_date.strftime('%Y-%m-%d')}

{email_body}
"""

            # Extract entity mentions
            mentions = self._extract_entity_mentions(email_text)

            # Save email
            email_path = self.output_dir / f"{email_id}.txt"
            with open(email_path, 'w') as f:
                f.write(email_text)

            # Add to metadata
            metadata['documents'].append({
                'id': email_id,
                'filename': f"{email_id}.txt",
                'type': 'unstructured',
                'source': 'generated',
                'timestamp': email_date.strftime('%Y-%m-%d'),
                'from': from_email,
                'to': to_email,
                'subject': subject,
                'entities_mentioned': mentions,
                'contradictions': contradictions,
                'confidence_alignment': 0.7 if not contradictions else 0.5
            })

            print(f"  Generated {email_id}: {subject[:50]}...")

        # Save metadata
        metadata_path = self.output_dir.parent / 'emails_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save name mapping
        mapping_path = self.output_dir.parent / 'name_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(self.name_mapping, f, indent=2)

        print(f"\n✅ Generated {num_emails} emails")
        print(f"   Saved to: {self.output_dir}")
        print(f"   Metadata: {metadata_path}")
        print(f"   Name mapping: {mapping_path}")

        return metadata


def main():
    generator = EmailGenerator()
    metadata = generator.generate_emails(num_emails=15)

    # Print summary
    print("\n" + "=" * 60)
    print("EMAIL GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total emails: {metadata['total_emails']}")

    total_contradictions = sum(len(doc['contradictions']) for doc in metadata['documents'])
    print(f"Total contradictions: {total_contradictions}")

    contradiction_types = {}
    for doc in metadata['documents']:
        for contradiction in doc['contradictions']:
            ctype = contradiction['type']
            contradiction_types[ctype] = contradiction_types.get(ctype, 0) + 1

    print("\nContradiction breakdown:")
    for ctype, count in contradiction_types.items():
        print(f"  {ctype}: {count}")

    print(
        f"\nAverage entity mentions per email: {sum(len(doc['entities_mentioned']) for doc in metadata['documents']) / len(metadata['documents']):.1f}")


if __name__ == '__main__':
    main()