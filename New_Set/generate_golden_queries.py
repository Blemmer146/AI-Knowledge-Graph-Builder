"""
Golden Query Generator - Fixed with Real Data
Generates 100-200 diverse, high-quality validation queries using actual CSV files

Query Types:
1. Basic facts (who, what, where)
2. Relationships (manager, team, projects)
3. Quantitative (how many, count, list all)
4. Temporal (when, recent, latest)
5. Comparative (vs, compared to, differences)
6. Aggregative (total, average, sum)
7. Negative (doesn't exist, not working on)
8. Complex multi-hop (X's manager's projects)
9. Semantic/external (GDPR, AWS, policies)
10. Contradiction detection

Author: Enterprise KG Project
Date: 2026-02-01 (Fixed)
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


class GoldenQueryGenerator:
    """Generate comprehensive golden queries for validation"""

    def __init__(self):
        self.queries = []
        self.query_id_counter = 1

        # Load ground truth data
        print("Loading actual CSV data...")
        self.employees = self._load_employees()
        self.projects = self._load_projects()
        self.products = self._load_products()
        self.policies = self._load_policies()
        self.project_assignments = self._load_project_assignments()

        print(f"  Loaded {len(self.employees)} employees")
        print(f"  Loaded {len(self.projects)} projects")
        print(f"  Loaded {len(self.products)} products")
        print(f"  Loaded {len(self.policies)} policies")
        print(f"  Loaded {len(self.project_assignments)} project assignments")

    def _load_employees(self) -> List[Dict]:
        """Load employee data from CSV"""
        try:
            df = pd.read_csv('data/source/structured/employees.csv')
            records = df.to_dict('records')

            # Normalize IDs
            for e in records:
                e['id'] = e['employee_id']

            return records
        except Exception as err:
            print(f"ERROR loading employees.csv: {err}")
            return []

    def _load_projects(self) -> List[Dict]:
        """Load project data from CSV"""
        try:
            df = pd.read_csv('data/source/structured/projects.csv')
            records = df.to_dict('records')

            # Normalize IDs
            for p in records:
                p['id'] = p['project_id']

            return records
        except Exception as err:
            print(f"ERROR loading projects.csv: {err}")
            return []

    def _load_products(self) -> List[Dict]:
        """Load product data from CSV"""
        try:
            df = pd.read_csv('data/source/structured/products.csv')
            records = df.to_dict('records')

            # Normalize IDs
            for p in records:
                p['id'] = p['product_id']

            return records
        except Exception as err:
            print(f"ERROR loading products.csv: {err}")
            return []

    def _load_policies(self) -> List[Dict]:
        """Load policy data from CSV"""
        try:
            df = pd.read_csv('data/source/structured/policies.csv')
            records = df.to_dict('records')

            # Normalize IDs
            for p in records:
                p['id'] = p['policy_id']

            return records
        except Exception as err:
            print(f"ERROR loading policies.csv: {err}")
            return []

    def _load_project_assignments(self) -> List[Dict]:
        """Load project assignments from CSV"""
        try:
            df = pd.read_csv('data/source/structured/project_assignments.csv')
            records = df.to_dict('records')

            # Normalize IDs
            for a in records:
                a['id'] = a['assignment_id']

            return records
        except Exception as err:
            print(f"ERROR loading project_assignments.csv: {err}")
            return []

    def generate_all_queries(self) -> List[Dict]:
        """Generate all query types"""
        print("\nGenerating comprehensive golden queries...")

        # 1. Basic fact queries (30-40)
        self._generate_basic_fact_queries()

        # 2. Relationship queries (25-30)
        self._generate_relationship_queries()

        # 3. Quantitative queries (15-20)
        self._generate_quantitative_queries()

        # 4. Temporal queries (10-15)
        self._generate_temporal_queries()

        # 5. Comparative queries (8-10)
        self._generate_comparative_queries()

        # 6. Aggregative queries (8-10)
        self._generate_aggregative_queries()

        # 7. Negative queries (8-10)
        self._generate_negative_queries()

        # 8. Complex multi-hop queries (10-15)
        self._generate_complex_queries()

        # 9. Semantic/external queries (10-12)
        self._generate_semantic_queries()

        # 10. Contradiction detection (5-8)
        self._generate_contradiction_queries()

        # 11. Edge case queries (5-8)
        self._generate_edge_case_queries()

        print(f"\nGenerated {len(self.queries)} total queries")
        return self.queries

    def _add_query(self, category: str, question: str, expected_answer: str,
                   expected_keywords: List[str] = None,
                   expected_entities: List[str] = None,
                   expected_source_files: List[str] = None,
                   should_succeed: bool = True,
                   confidence_threshold: float = 0.7,
                   max_latency_ms: int = 3000,
                   should_surface_contradiction: bool = False,
                   query_type: str = None,
                   notes: str = None):
        """Add query to list"""
        query_id = f"{category}_{self.query_id_counter:03d}"
        self.query_id_counter += 1

        query = {
            'id': query_id,
            'question': question,
            'category': category,
            'expected_answer': expected_answer,
            'should_succeed': should_succeed,
            'confidence_threshold': confidence_threshold,
            'max_latency_ms': max_latency_ms
        }

        if expected_keywords:
            query['expected_keywords'] = expected_keywords
        if expected_entities:
            query['expected_entities'] = expected_entities
        if expected_source_files:
            query['expected_source_files'] = expected_source_files
        if should_surface_contradiction:
            query['should_surface_contradiction'] = should_surface_contradiction
        if query_type:
            query['query_type'] = query_type
        if notes:
            query['notes'] = notes

        self.queries.append(query)

    def _generate_basic_fact_queries(self):
        """Generate basic who/what/where queries"""
        print("  Generating basic fact queries...")

        # Manager queries for all employees with managers
        for emp in self.employees:
            if emp.get('manager_id') and pd.notna(emp.get('manager_id')):
                manager = next((e for e in self.employees if e['id'] == emp['manager_id']), None)
                if manager:
                    self._add_query(
                        category='basic_fact',
                        question=f"Who is {emp['full_name']}'s manager?",
                        expected_answer=manager['full_name'],
                        expected_keywords=[manager['full_name'], 'manager'],
                        expected_entities=[emp['id'], manager['id']],
                        expected_source_files=['employees.csv'],
                        confidence_threshold=0.85,
                        max_latency_ms=2000,
                        query_type='manager_lookup'
                    )

        # Department queries for all employees
        for emp in self.employees[:20]:  # Limit to avoid too many
            if emp.get('department'):
                self._add_query(
                    category='basic_fact',
                    question=f"What department does {emp['full_name']} work in?",
                    expected_answer=emp['department'],
                    expected_keywords=[emp['department']],
                    expected_entities=[emp['id']],
                    expected_source_files=['employees.csv'],
                    confidence_threshold=0.85,
                    max_latency_ms=2000,
                    query_type='department_lookup'
                )

        # Role queries for all employees
        for emp in self.employees[:15]:  # Limit to avoid too many
            if emp.get('role'):
                self._add_query(
                    category='basic_fact',
                    question=f"What is {emp['full_name']}'s role?",
                    expected_answer=emp['role'],
                    expected_keywords=[emp['role']],
                    expected_entities=[emp['id']],
                    expected_source_files=['employees.csv'],
                    confidence_threshold=0.85,
                    max_latency_ms=2000,
                    query_type='role_lookup'
                )

    def _generate_relationship_queries(self):
        """Generate relationship queries"""
        print("  Generating relationship queries...")

        # Who works on project X? (for all projects)
        for proj in self.projects:
            team_members = [a for a in self.project_assignments if a['project_id'] == proj['id']]
            if team_members:
                member_names = []
                member_ids = []
                for assignment in team_members[:5]:  # Limit to top 5
                    emp = next((e for e in self.employees if e['id'] == assignment['employee_id']), None)
                    if emp:
                        member_names.append(emp['full_name'])
                        member_ids.append(emp['id'])

                if member_names:
                    self._add_query(
                        category='relationship',
                        question=f"Who works on {proj['name']}?",
                        expected_answer=f"{', '.join(member_names)}",
                        expected_keywords=member_names[:3],  # At least first 3 names
                        expected_entities=[proj['id']] + member_ids,
                        expected_source_files=['project_assignments.csv'],
                        confidence_threshold=0.75,
                        max_latency_ms=2500,
                        query_type='project_team'
                    )

        # What projects does X work on? (for employees with projects)
        for emp in self.employees:
            emp_projects = [a for a in self.project_assignments if a['employee_id'] == emp['id']]
            if emp_projects:
                project_names = []
                project_ids = []
                for assignment in emp_projects:
                    proj = next((p for p in self.projects if p['id'] == assignment['project_id']), None)
                    if proj:
                        project_names.append(proj['name'])
                        project_ids.append(proj['id'])

                if project_names:
                    self._add_query(
                        category='relationship',
                        question=f"What projects does {emp['full_name']} work on?",
                        expected_answer=f"{', '.join(project_names)}",
                        expected_keywords=project_names[:2],
                        expected_entities=[emp['id']] + project_ids,
                        expected_source_files=['project_assignments.csv'],
                        confidence_threshold=0.75,
                        max_latency_ms=2500,
                        query_type='employee_projects'
                    )

        # Who manages the X department?
        departments = list(set([e['department'] for e in self.employees if e.get('department')]))
        for dept in departments:
            # Find department manager (someone with "Manager" in role and matches department)
            dept_managers = [e for e in self.employees
                           if e.get('department') == dept
                           and ('Manager' in e.get('role', '') or 'Officer' in e.get('role', ''))]

            if dept_managers:
                manager = dept_managers[0]
                self._add_query(
                    category='relationship',
                    question=f"Who manages the {dept} department?",
                    expected_answer=manager['full_name'],
                    expected_keywords=[manager['full_name'], dept],
                    expected_entities=[manager['id']],
                    expected_source_files=['employees.csv'],
                    confidence_threshold=0.70,
                    max_latency_ms=2500,
                    query_type='department_manager'
                )

    def _generate_quantitative_queries(self):
        """Generate quantitative queries"""
        print("  Generating quantitative queries...")

        # How many employees work in X department?
        departments = list(set([e['department'] for e in self.employees if e.get('department')]))
        for dept in departments:
            count = len([e for e in self.employees if e.get('department') == dept])
            self._add_query(
                category='quantitative',
                question=f"How many employees work in the {dept} department?",
                expected_answer=str(count),
                expected_keywords=[str(count), dept, 'employee'],
                expected_source_files=['employees.csv'],
                confidence_threshold=0.70,
                max_latency_ms=3000,
                query_type='count_employees'
            )

        # How many projects is X working on?
        for emp in self.employees[:10]:
            project_count = len([a for a in self.project_assignments if a['employee_id'] == emp['id']])
            if project_count > 0:
                self._add_query(
                    category='quantitative',
                    question=f"How many projects is {emp['full_name']} working on?",
                    expected_answer=str(project_count),
                    expected_keywords=[str(project_count), 'project'],
                    expected_entities=[emp['id']],
                    expected_source_files=['project_assignments.csv'],
                    confidence_threshold=0.70,
                    max_latency_ms=3000,
                    query_type='count_projects'
                )

        # List all employees in X department
        for dept in departments[:5]:
            dept_employees = [e for e in self.employees if e.get('department') == dept]
            employee_names = [e['full_name'] for e in dept_employees[:5]]
            self._add_query(
                category='quantitative',
                question=f"List all employees in the {dept} department",
                expected_answer=f"{', '.join([e['full_name'] for e in dept_employees])}",
                expected_keywords=employee_names[:3],
                expected_source_files=['employees.csv'],
                confidence_threshold=0.65,
                max_latency_ms=3500,
                query_type='list_employees'
            )

        # How many people report to X?
        for emp in self.employees:
            direct_reports = [e for e in self.employees if e.get('manager_id') == emp['id']]
            if direct_reports:
                self._add_query(
                    category='quantitative',
                    question=f"How many people report to {emp['full_name']}?",
                    expected_answer=str(len(direct_reports)),
                    expected_keywords=[str(len(direct_reports)), 'report'],
                    expected_entities=[emp['id']],
                    expected_source_files=['employees.csv'],
                    confidence_threshold=0.70,
                    max_latency_ms=3000,
                    query_type='count_reports'
                )

    def _generate_temporal_queries(self):
        """Generate temporal queries"""
        print("  Generating temporal queries...")

        # When was project X started?
        for proj in self.projects[:8]:
            if proj.get('start_date'):
                self._add_query(
                    category='temporal',
                    question=f"When was {proj['name']} started?",
                    expected_answer=proj['start_date'],
                    expected_keywords=[proj['name'], 'start', proj['start_date']],
                    expected_entities=[proj['id']],
                    expected_source_files=['projects.csv'],
                    confidence_threshold=0.70,
                    max_latency_ms=3000,
                    query_type='project_timeline'
                )

        # What is the status of project X?
        for proj in self.projects:
            if proj.get('status'):
                self._add_query(
                    category='temporal',
                    question=f"What is the status of {proj['name']}?",
                    expected_answer=proj['status'],
                    expected_keywords=[proj['status'], proj['name']],
                    expected_entities=[proj['id']],
                    expected_source_files=['projects.csv'],
                    confidence_threshold=0.75,
                    max_latency_ms=2500,
                    query_type='project_status'
                )

        # Which projects are active?
        active_projects = [p for p in self.projects if p.get('status') == 'Active']
        if active_projects:
            project_names = [p['name'] for p in active_projects]
            self._add_query(
                category='temporal',
                question="Which projects are currently active?",
                expected_answer=f"{', '.join(project_names)}",
                expected_keywords=project_names[:3] + ['active'],
                expected_source_files=['projects.csv'],
                confidence_threshold=0.70,
                max_latency_ms=3000,
                query_type='active_projects'
            )

        # Which projects are completed?
        completed_projects = [p for p in self.projects if p.get('status') == 'Completed']
        if completed_projects:
            project_names = [p['name'] for p in completed_projects[:5]]
            self._add_query(
                category='temporal',
                question="Which projects have been completed?",
                expected_answer=f"{', '.join([p['name'] for p in completed_projects])}",
                expected_keywords=project_names[:3] + ['completed'],
                expected_source_files=['projects.csv'],
                confidence_threshold=0.70,
                max_latency_ms=3000,
                query_type='completed_projects'
            )

    def _generate_comparative_queries(self):
        """Generate comparative queries"""
        print("  Generating comparative queries...")

        # X vs Y department size
        departments = list(set([e['department'] for e in self.employees if e.get('department')]))
        if len(departments) >= 2:
            for i in range(min(4, len(departments) - 1)):
                dept1, dept2 = departments[i], departments[i + 1]
                count1 = len([e for e in self.employees if e.get('department') == dept1])
                count2 = len([e for e in self.employees if e.get('department') == dept2])

                self._add_query(
                    category='comparative',
                    question=f"How does {dept1} compare to {dept2} in team size?",
                    expected_answer=f"{dept1} has {count1} employees, {dept2} has {count2} employees",
                    expected_keywords=[dept1, dept2, str(count1), str(count2)],
                    expected_source_files=['employees.csv'],
                    confidence_threshold=0.60,
                    max_latency_ms=3500,
                    query_type='department_comparison'
                )

        # Who has more team members: X or Y?
        managers = [e for e in self.employees if 'Manager' in e.get('role', '')]
        if len(managers) >= 2:
            for i in range(min(3, len(managers) - 1)):
                mgr1, mgr2 = managers[i], managers[i + 1]
                team1 = [e for e in self.employees if e.get('manager_id') == mgr1['id']]
                team2 = [e for e in self.employees if e.get('manager_id') == mgr2['id']]

                self._add_query(
                    category='comparative',
                    question=f"Who has more direct reports: {mgr1['full_name']} or {mgr2['full_name']}?",
                    expected_answer=f"{mgr1['full_name'] if len(team1) >= len(team2) else mgr2['full_name']}",
                    expected_keywords=[mgr1['full_name'], mgr2['full_name'], 'report'],
                    expected_entities=[mgr1['id'], mgr2['id']],
                    expected_source_files=['employees.csv'],
                    confidence_threshold=0.60,
                    max_latency_ms=3500,
                    query_type='manager_comparison'
                )

    def _generate_aggregative_queries(self):
        """Generate aggregative queries"""
        print("  Generating aggregative queries...")

        # Total number of employees
        self._add_query(
            category='aggregative',
            question="How many total employees does CodeFlow have?",
            expected_answer=str(len(self.employees)),
            expected_keywords=[str(len(self.employees)), 'employee', 'total'],
            expected_source_files=['employees.csv'],
            confidence_threshold=0.70,
            max_latency_ms=3000,
            query_type='total_count'
        )

        # Total number of projects
        self._add_query(
            category='aggregative',
            question="How many total projects does CodeFlow have?",
            expected_answer=str(len(self.projects)),
            expected_keywords=[str(len(self.projects)), 'project'],
            expected_source_files=['projects.csv'],
            confidence_threshold=0.70,
            max_latency_ms=3000,
            query_type='total_projects'
        )

        # All departments
        departments = list(set([e['department'] for e in self.employees if e.get('department')]))
        self._add_query(
            category='aggregative',
            question="What are all the departments at CodeFlow?",
            expected_answer=f"{', '.join(sorted(departments))}",
            expected_keywords=departments[:3],
            expected_source_files=['employees.csv'],
            confidence_threshold=0.70,
            max_latency_ms=3000,
            query_type='list_departments'
        )

        # All managers
        managers = [e['full_name'] for e in self.employees if 'Manager' in e.get('role', '')]
        self._add_query(
            category='aggregative',
            question="Who are all the managers at CodeFlow?",
            expected_answer=f"{', '.join(managers)}",
            expected_keywords=managers[:3],
            expected_source_files=['employees.csv'],
            confidence_threshold=0.70,
            max_latency_ms=3500,
            query_type='list_managers'
        )

        # All products
        product_names = [p['name'] for p in self.products]
        self._add_query(
            category='aggregative',
            question="What products does CodeFlow use?",
            expected_answer=f"{', '.join(product_names)}",
            expected_keywords=product_names[:3],
            expected_source_files=['products.csv'],
            confidence_threshold=0.70,
            max_latency_ms=3000,
            query_type='list_products'
        )

    def _generate_negative_queries(self):
        """Generate negative/non-existent queries"""
        print("  Generating negative queries...")

        # Non-existent person
        fake_names = ["Elon Musk", "Jeff Bezos", "Tim Cook", "Satya Nadella", "Mark Zuckerberg"]
        for name in fake_names:
            self._add_query(
                category='negative',
                question=f"Who is {name}'s manager at CodeFlow?",
                expected_answer="N/A",
                should_succeed=False,
                confidence_threshold=0.0,
                max_latency_ms=2500,
                query_type='non_existent_person',
                notes=f'{name} does not exist in database'
            )

        # Non-existent project
        fake_projects = ["Project Unicorn", "Project Dragon", "Project Phoenix", "Project Nebula"]
        for proj in fake_projects:
            self._add_query(
                category='negative',
                question=f"Who works on {proj}?",
                expected_answer="N/A",
                should_succeed=False,
                confidence_threshold=0.0,
                max_latency_ms=2500,
                query_type='non_existent_project',
                notes=f'{proj} does not exist in database'
            )

        # Employee not working on any project
        no_project_employees = [e for e in self.employees if not any(
            a['employee_id'] == e['id'] for a in self.project_assignments
        )]
        if no_project_employees:
            for emp in no_project_employees[:3]:
                self._add_query(
                    category='negative',
                    question=f"What projects is {emp['full_name']} working on?",
                    expected_answer=f"{emp['full_name']} is not assigned to any projects",
                    expected_keywords=[emp['full_name']],
                    expected_entities=[emp['id']],
                    expected_source_files=['project_assignments.csv', 'employees.csv'],
                    confidence_threshold=0.60,
                    max_latency_ms=3000,
                    query_type='no_projects'
                )

    def _generate_complex_queries(self):
        """Generate complex multi-hop queries"""
        print("  Generating complex queries...")

        # X's manager's projects
        for emp in self.employees[:8]:
            if emp.get('manager_id') and pd.notna(emp.get('manager_id')):
                manager = next((e for e in self.employees if e['id'] == emp['manager_id']), None)
                if manager:
                    mgr_projects = [a for a in self.project_assignments if a['employee_id'] == manager['id']]
                    if mgr_projects:
                        proj_names = []
                        proj_ids = []
                        for assignment in mgr_projects:
                            proj = next((p for p in self.projects if p['id'] == assignment['project_id']), None)
                            if proj:
                                proj_names.append(proj['name'])
                                proj_ids.append(proj['id'])

                        if proj_names:
                            self._add_query(
                                category='complex',
                                question=f"What projects does {emp['full_name']}'s manager work on?",
                                expected_answer=f"{manager['full_name']} works on {', '.join(proj_names)}",
                                expected_keywords=[manager['full_name']] + proj_names[:2],
                                expected_entities=[emp['id'], manager['id']] + proj_ids,
                                expected_source_files=['employees.csv', 'project_assignments.csv'],
                                confidence_threshold=0.60,
                                max_latency_ms=4000,
                                query_type='manager_projects'
                            )

        # Team members of X's projects
        for emp in self.employees[:8]:
            emp_projects = [a for a in self.project_assignments if a['employee_id'] == emp['id']]
            if emp_projects:
                proj = next((p for p in self.projects if p['id'] == emp_projects[0]['project_id']), None)
                if proj:
                    team = [a for a in self.project_assignments
                           if a['project_id'] == proj['id'] and a['employee_id'] != emp['id']]
                    if team:
                        team_names = []
                        team_ids = []
                        for assignment in team[:4]:
                            teammate = next((e for e in self.employees if e['id'] == assignment['employee_id']), None)
                            if teammate:
                                team_names.append(teammate['full_name'])
                                team_ids.append(teammate['id'])

                        if team_names:
                            self._add_query(
                                category='complex',
                                question=f"Who else works on {emp['full_name']}'s projects?",
                                expected_answer=f"Team members: {', '.join(team_names)}",
                                expected_keywords=team_names[:2],
                                expected_entities=[emp['id'], proj['id']] + team_ids,
                                expected_source_files=['project_assignments.csv'],
                                confidence_threshold=0.60,
                                max_latency_ms=4000,
                                query_type='project_teammates'
                            )

        # Department's project involvement
        departments = list(set([e['department'] for e in self.employees if e.get('department')]))
        for dept in departments:
            dept_employees = [e['id'] for e in self.employees if e.get('department') == dept]
            dept_projects_ids = list(set([a['project_id'] for a in self.project_assignments
                                         if a['employee_id'] in dept_employees]))
            if dept_projects_ids:
                proj_names = [p['name'] for p in self.projects if p['id'] in dept_projects_ids]
                if proj_names:
                    self._add_query(
                        category='complex',
                        question=f"What projects involve people from the {dept} department?",
                        expected_answer=f"{dept} department works on: {', '.join(proj_names)}",
                        expected_keywords=[dept] + proj_names[:2],
                        expected_source_files=['employees.csv', 'project_assignments.csv'],
                        confidence_threshold=0.60,
                        max_latency_ms=4000,
                        query_type='department_projects'
                    )

    def _generate_semantic_queries(self):
        """Generate semantic/external queries"""
        print("  Generating semantic queries...")

        # GDPR questions
        gdpr_questions = [
            ("What does GDPR say about data retention?",
             "GDPR requires data to be retained only as long as necessary",
             ['retention', 'data', 'GDPR', 'necessary']),
            ("How should we handle customer data according to GDPR?",
             "Customer data must be processed securely with appropriate safeguards",
             ['customer', 'data', 'GDPR', 'secure']),
            ("What are the GDPR requirements for data processing?",
             "Data processing requires lawful basis, transparency, and security measures",
             ['GDPR', 'processing', 'lawful', 'security']),
        ]

        for question, answer, keywords in gdpr_questions:
            self._add_query(
                category='semantic',
                question=question,
                expected_answer=answer,
                expected_keywords=keywords,
                expected_source_files=['GDPR_summary.pdf'],
                confidence_threshold=0.60,
                max_latency_ms=4000,
                query_type='gdpr'
            )

        # AWS questions
        aws_questions = [
            ("What are AWS best practices for security?",
             "AWS recommends encryption, access controls, monitoring, and compliance",
             ['AWS', 'security', 'encryption', 'monitoring']),
            ("How does AWS handle compliance?",
             "AWS provides compliance certifications and shared responsibility model",
             ['AWS', 'compliance', 'certification']),
        ]

        for question, answer, keywords in aws_questions:
            self._add_query(
                category='semantic',
                question=question,
                expected_answer=answer,
                expected_keywords=keywords,
                expected_source_files=['AWS_compliance.pdf'],
                confidence_threshold=0.60,
                max_latency_ms=4000,
                query_type='aws'
            )

        # Policy ownership (using actual data)
        for policy in self.policies:
            if policy.get('owner_id'):
                owner = next((e for e in self.employees if e['id'] == policy['owner_id']), None)
                if owner:
                    self._add_query(
                        category='semantic',
                        question=f"Who owns the {policy['name']}?",
                        expected_answer=owner['full_name'],
                        expected_keywords=[owner['full_name'], policy['name']],
                        expected_entities=[policy['id'], owner['id']],
                        expected_source_files=['policies.csv'],
                        confidence_threshold=0.75,
                        max_latency_ms=2500,
                        query_type='policy_owner'
                    )

    def _generate_contradiction_queries(self):
        """Generate contradiction detection queries"""
        print("  Generating contradiction queries...")

        # These should detect contradictions between CSV and documents
        # Note: You'll need to verify actual contradictions in your data
        self._add_query(
            category='contradiction',
            question="What project is Debra Gardner working on?",
            expected_answer="Debra Gardner works on Jira Cloud Adoption and Identity & Access Management Refresh",
            expected_keywords=['Debra Gardner', 'Jira Cloud Adoption', 'Identity'],
            expected_entities=['emp_008'],
            confidence_threshold=0.70,
            max_latency_ms=3500,
            query_type='project_lookup',
            notes='Check for potential contradictions in documents'
        )

        self._add_query(
            category='contradiction',
            question="Tell me about the Jira Cloud Adoption project",
            expected_answer="Jira Cloud Adoption project details",
            expected_keywords=['Jira', 'Cloud', 'Adoption'],
            expected_entities=['proj_001'],
            expected_source_files=['projects.csv'],
            confidence_threshold=0.65,
            max_latency_ms=3500,
            query_type='project_info'
        )

    def _generate_edge_case_queries(self):
        """Generate edge case queries"""
        print("  Generating edge case queries...")

        # Very broad queries
        self._add_query(
            category='edge_case',
            question="Tell me about CodeFlow",
            expected_answer="General information about CodeFlow Corp",
            expected_keywords=['CodeFlow', 'company', 'employee'],
            confidence_threshold=0.50,
            max_latency_ms=4000,
            query_type='broad_query'
        )

        # Ambiguous person reference (using actual CEO)
        ceo = next((e for e in self.employees if 'Chief Executive Officer' in e.get('role', '')), None)
        if ceo:
            first_name = ceo['full_name'].split()[0]
            self._add_query(
                category='edge_case',
                question=f"What does {first_name} do?",
                expected_answer=f"{ceo['full_name']} is the {ceo['role']}",
                expected_keywords=[ceo['full_name'], ceo['role']],
                expected_entities=[ceo['id']],
                expected_source_files=['employees.csv'],
                confidence_threshold=0.65,
                max_latency_ms=2500,
                query_type='ambiguous_reference'
            )

        # Multiple interpretations
        self._add_query(
            category='edge_case',
            question="Who is responsible for security?",
            expected_answer="Information about security responsibilities",
            expected_keywords=['security', 'responsible'],
            confidence_threshold=0.50,
            max_latency_ms=3500,
            query_type='multiple_meanings'
        )

        # Partial information (using actual project)
        if self.projects:
            proj = self.projects[0]
            self._add_query(
                category='edge_case',
                question=proj['name'],
                expected_answer=f"Information about {proj['name']}",
                expected_keywords=[proj['name']],
                expected_entities=[proj['id']],
                confidence_threshold=0.60,
                max_latency_ms=3000,
                query_type='incomplete_query'
            )

    def save_queries(self, output_path: str = "golden_queries.json"):
        """Save queries to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.queries, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved {len(self.queries)} queries to {output_path}")

        # Print statistics
        categories = {}
        for query in self.queries:
            cat = query['category']
            categories[cat] = categories.get(cat, 0) + 1

        print("\nQuery Distribution:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Golden Query Generator - Using Real Data")
    print("=" * 60)

    generator = GoldenQueryGenerator()

    if not generator.employees:
        print("\nERROR: No data loaded. Check that CSV files exist in data/source/structured/")
        return

    queries = generator.generate_all_queries()
    generator.save_queries()

    print("\n✓ Query generation complete!")


if __name__ == "__main__":
    main()