"""
Phase 1: Structured Data Generation for CodeFlow Knowledge Graph

This script generates the ground truth structured data (CSV files) and the
canonical entity registry (entities.json) that serves as the foundation for
the entire knowledge graph project.

Key Design Principles:
- Structured data = 100% consistent (confidence 1.0)
- All entities have opaque IDs (emp_001, proj_001, etc.)
- Referential integrity enforced
- No duplicate names (e.g., only ONE "Alice")
"""

import pandas as pd
import json
import yaml
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from faker import Faker

# Initialize Faker with seed for reproducibility
fake = Faker()


class StructuredDataGenerator:
    """Generates all structured data and entity registry for CodeFlow."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed for reproducibility
        random.seed(self.config['generation']['random_seed'])
        Faker.seed(self.config['generation']['random_seed'])

        # Create output directories
        self._create_directories()

        # Track used names to avoid duplicates
        self.used_first_names = set()
        self.used_full_names = set()

        print("✓ Configuration loaded")
        print(f"✓ Random seed: {self.config['generation']['random_seed']}")

    def _create_directories(self):
        """Create output directory structure."""
        for path_key in ['output_dir', 'structured_dir', 'semi_structured_dir',
                         'unstructured_dir', 'external_dir']:
            Path(self.config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
        print("✓ Directory structure created")

    def _generate_unique_name(self) -> Tuple[str, str, str]:
        """
        Generate a unique name ensuring no duplicate first names.

        Returns:
            Tuple of (full_name, first_name, last_name)
        """
        max_attempts = 100
        for _ in range(max_attempts):
            first_name = fake.first_name()
            last_name = fake.last_name()
            full_name = f"{first_name} {last_name}"

            # Check for duplicates
            if first_name not in self.used_first_names and \
                    full_name not in self.used_full_names:
                self.used_first_names.add(first_name)
                self.used_full_names.add(full_name)
                return full_name, first_name, last_name

        raise ValueError("Could not generate unique name after maximum attempts")

    def _random_date_in_range(self, start: str, end: str) -> str:
        """Generate random date in YYYY-MM-DD format within range."""
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        random_date = start_date + timedelta(days=random_days)

        return random_date.strftime("%Y-%m-%d")

    def _random_year_month(self, start: str, end: str) -> str:
        """Generate random year-month in YYYY-MM format."""
        date_str = self._random_date_in_range(
            f"{start}-01",
            f"{end}-01"
        )
        return date_str[:7]  # Return YYYY-MM only

    def generate_employees(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate employee data with organizational hierarchy.

        Returns:
            Tuple of (DataFrame, entity_list)
        """
        print("\n[1/5] Generating employees...")

        num_employees = self.config['entities']['num_employees']
        departments = self.config['departments']
        company_domain = self.config['company']['domain']

        employees = []
        entity_list = []

        # Generate CEO first (no manager)
        full_name, first_name, last_name = self._generate_unique_name()
        emp_id = "emp_001"

        emp_data = {
            'employee_id': emp_id,
            'full_name': full_name,
            'role': 'Chief Executive Officer',
            'department': 'Executive',
            'manager_id': None,
            'hire_date': self._random_date_in_range(
                self.config['employees']['hire_date_range']['start'],
                "2019-12-31"  # CEO hired early
            ),
            'email': f"{first_name.lower()}.{last_name.lower()}@{company_domain}"
        }
        employees.append(emp_data)

        entity_list.append({
            'id': emp_id,
            'full_name': full_name,
            'first_name': first_name,
            'last_name': last_name,
            'role': emp_data['role'],
            'department': emp_data['department'],
            'email': emp_data['email']
        })

        # Generate department heads
        dept_heads = {}
        emp_counter = 2

        for dept in departments:
            full_name, first_name, last_name = self._generate_unique_name()
            emp_id = f"emp_{emp_counter:03d}"

            # Determine role based on department
            if dept == 'Engineering':
                role = 'Engineering Manager'
            elif dept == 'Legal':
                role = 'Legal Counsel'
            elif dept == 'Operations':
                role = 'Operations Manager'
            elif dept == 'Sales':
                role = 'Sales Manager'
            elif dept == 'Finance':
                role = 'Finance Manager'
            else:
                role = f"{dept} Manager"

            emp_data = {
                'employee_id': emp_id,
                'full_name': full_name,
                'role': role,
                'department': dept,
                'manager_id': 'emp_001',  # Reports to CEO
                'hire_date': self._random_date_in_range(
                    self.config['employees']['hire_date_range']['start'],
                    self.config['employees']['hire_date_range']['end']
                ),
                'email': f"{first_name.lower()}.{last_name.lower()}@{company_domain}"
            }
            employees.append(emp_data)

            entity_list.append({
                'id': emp_id,
                'full_name': full_name,
                'first_name': first_name,
                'last_name': last_name,
                'role': emp_data['role'],
                'department': dept,
                'email': emp_data['email']
            })

            dept_heads[dept] = emp_id
            emp_counter += 1

        # Generate remaining employees
        remaining = num_employees - len(employees)

        for i in range(remaining):
            emp_id = f"emp_{emp_counter:03d}"
            full_name, first_name, last_name = self._generate_unique_name()

            # Assign to random department
            dept = random.choice(departments)

            # Select role from department role pool
            roles = self.config['employees']['roles_by_department'][dept]
            # Avoid manager roles (already assigned)
            non_manager_roles = [r for r in roles if 'Manager' not in r]
            role = random.choice(non_manager_roles)

            emp_data = {
                'employee_id': emp_id,
                'full_name': full_name,
                'role': role,
                'department': dept,
                'manager_id': dept_heads[dept],  # Reports to department head
                'hire_date': self._random_date_in_range(
                    self.config['employees']['hire_date_range']['start'],
                    self.config['employees']['hire_date_range']['end']
                ),
                'email': f"{first_name.lower()}.{last_name.lower()}@{company_domain}"
            }
            employees.append(emp_data)

            entity_list.append({
                'id': emp_id,
                'full_name': full_name,
                'first_name': first_name,
                'last_name': last_name,
                'role': role,
                'department': dept,
                'email': emp_data['email']
            })

            emp_counter += 1

        df = pd.DataFrame(employees)
        print(f"  ✓ Generated {len(df)} employees across {len(departments) + 1} departments")
        print(f"  ✓ All first names unique: {len(self.used_first_names)} names")

        return df, entity_list

    def generate_projects(self, employees_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate project data.

        Args:
            employees_df: Employee DataFrame for department alignment

        Returns:
            Tuple of (DataFrame, entity_list)
        """
        print("\n[2/5] Generating projects...")

        num_projects = self.config['entities']['num_projects']
        project_pool = self.config['project_names']
        status_dist = self.config['projects']['status_distribution']

        # Sample project names without replacement
        selected_names = random.sample(project_pool, num_projects)

        projects = []
        entity_list = []

        # Project description templates
        descriptions = [
            "Cloud migration initiative to move legacy systems to AWS infrastructure",
            "Customer data platform consolidation for improved analytics",
            "Security compliance automation for regulatory requirements",
            "Sales pipeline optimization using machine learning",
            "Internal developer portal for improved productivity",
            "Real-time monitoring dashboard for operational metrics",
            "API gateway modernization for microservices architecture",
            "Data warehouse migration to support advanced analytics",
            "Mobile application development for customer engagement",
            "Infrastructure automation using infrastructure-as-code",
            "Customer onboarding workflow automation",
            "Enterprise search platform implementation",
        ]

        for i, name in enumerate(selected_names, 1):
            proj_id = f"proj_{i:03d}"

            # Determine status based on distribution
            rand_val = random.random()
            if rand_val < status_dist['Completed']:
                status = 'Completed'
            elif rand_val < status_dist['Completed'] + status_dist['Active']:
                status = 'Active'
            else:
                status = 'Planned'

            # Assign to department (weighted toward Engineering)
            if random.random() < 0.6:
                dept = 'Engineering'
            else:
                dept = random.choice([d for d in self.config['departments']
                                      if d != 'Executive'])

            # Generate dates based on status
            start = self._random_year_month(
                self.config['time_range']['start'],
                "2024-06"
            )

            if status == 'Completed':
                # End date in the past
                end = self._random_year_month(
                    start,
                    "2024-12"
                )
            elif status == 'Active':
                # End date in the future
                end = self._random_year_month(
                    "2024-12",
                    self.config['time_range']['end']
                )
            else:  # Planned
                # Both dates in future
                start = self._random_year_month("2024-09", "2025-01")
                end = self._random_year_month(start, "2025-12")

            proj_data = {
                'project_id': proj_id,
                'name': name,
                'description': descriptions[i - 1],
                'status': status,
                'start_date': start,
                'end_date': end,
                'department': dept
            }
            projects.append(proj_data)

            entity_list.append({
                'id': proj_id,
                'name': name,
                'description': proj_data['description'],
                'status': status,
                'department': dept
            })

        df = pd.DataFrame(projects)
        print(f"  ✓ Generated {len(df)} projects")
        print(f"  ✓ Status breakdown: {df['status'].value_counts().to_dict()}")

        return df, entity_list

    def generate_products(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate product/vendor data.

        Returns:
            Tuple of (DataFrame, entity_list)
        """
        print("\n[3/5] Generating products...")

        product_configs = self.config['products']

        products = []
        entity_list = []

        for i, prod_config in enumerate(product_configs, 1):
            prod_id = f"prod_{i:03d}"

            # Generate realistic acquisition dates and costs
            acquisition_date = self._random_year_month(
                self.config['time_range']['start'],
                "2024-06"
            )

            # Cost varies by category
            if prod_config['category'] == 'Cloud':
                cost = random.randint(100000, 500000)
            elif prod_config['category'] == 'CRM':
                cost = random.randint(50000, 200000)
            else:
                cost = random.randint(10000, 100000)

            prod_data = {
                'product_id': prod_id,
                'name': prod_config['name'],
                'vendor': prod_config['vendor'],
                'category': prod_config['category'],
                'acquisition_date': acquisition_date,
                'annual_cost': cost
            }
            products.append(prod_data)

            entity_list.append({
                'id': prod_id,
                'name': prod_config['name'],
                'vendor': prod_config['vendor'],
                'category': prod_config['category']
            })

        df = pd.DataFrame(products)
        print(f"  ✓ Generated {len(df)} products")
        print(f"  ✓ Categories: {df['category'].value_counts().to_dict()}")

        return df, entity_list

    def generate_policies(self, employees_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Generate internal policy data.

        Args:
            employees_df: Employee DataFrame for owner assignment

        Returns:
            Tuple of (DataFrame, entity_list)
        """
        print("\n[4/5] Generating policies...")

        policy_configs = self.config['policies']

        policies = []
        entity_list = []

        for i, pol_config in enumerate(policy_configs, 1):
            pol_id = f"pol_{i:03d}"

            # Assign owner (prefer Legal or Operations employees)
            dept_filter = employees_df['department'].isin(['Legal', 'Operations', 'Executive'])
            potential_owners = employees_df[dept_filter]['employee_id'].tolist()
            owner_id = random.choice(potential_owners)

            # Version number
            version = f"{random.randint(1, 3)}.{random.randint(0, 5)}"

            # Effective date
            effective_date = self._random_year_month(
                self.config['time_range']['start'],
                "2024-12"
            )

            pol_data = {
                'policy_id': pol_id,
                'name': pol_config['name'],
                'category': pol_config['category'],
                'version': version,
                'effective_date': effective_date,
                'owner_id': owner_id
            }
            policies.append(pol_data)

            entity_list.append({
                'id': pol_id,
                'name': pol_config['name'],
                'category': pol_config['category'],
                'version': version
            })

        df = pd.DataFrame(policies)
        print(f"  ✓ Generated {len(df)} policies")

        return df, entity_list

    def generate_project_assignments(self,
                                     employees_df: pd.DataFrame,
                                     projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate project assignments linking employees to projects.

        Args:
            employees_df: Employee DataFrame
            projects_df: Project DataFrame

        Returns:
            DataFrame of assignments
        """
        print("\n[5/5] Generating project assignments...")

        assignments = []
        assignment_counter = 1

        for _, project in projects_df.iterrows():
            # Each project gets 2-5 people assigned
            num_assignments = random.randint(2, 5)

            # Filter employees from same department as project
            dept_employees = employees_df[
                employees_df['department'] == project['department']
                ]['employee_id'].tolist()

            # If not enough in department, add from other departments
            if len(dept_employees) < num_assignments:
                other_employees = employees_df[
                    employees_df['department'] != project['department']
                    ]['employee_id'].tolist()
                dept_employees.extend(random.sample(
                    other_employees,
                    min(num_assignments - len(dept_employees), len(other_employees))
                ))

            # Sample employees for this project
            assigned_employees = random.sample(
                dept_employees,
                min(num_assignments, len(dept_employees))
            )

            # Assign roles
            roles = self.config['assignments']['roles']
            allocations = self.config['assignments']['allocation_options']

            for i, emp_id in enumerate(assigned_employees):
                asgn_id = f"asgn_{assignment_counter:03d}"

                # First person is often the lead
                if i == 0:
                    role = random.choice(['Lead Engineer', 'Project Manager', 'Technical Lead'])
                    allocation = random.choice([75, 100])
                else:
                    role = random.choice([r for r in roles if 'Lead' not in r and 'Manager' not in r])
                    allocation = random.choice(allocations)

                # Start date same as project or slightly after
                start_date = project['start_date']

                asgn_data = {
                    'assignment_id': asgn_id,
                    'employee_id': emp_id,
                    'project_id': project['project_id'],
                    'role': role,
                    'allocation_pct': allocation,
                    'start_date': start_date
                }
                assignments.append(asgn_data)
                assignment_counter += 1

        df = pd.DataFrame(assignments)
        print(f"  ✓ Generated {len(df)} assignments")
        print(f"  ✓ Average {len(df) / len(projects_df):.1f} people per project")

        return df

    def generate_entities_json(self,
                               employee_entities: List[Dict],
                               project_entities: List[Dict],
                               product_entities: List[Dict],
                               policy_entities: List[Dict]) -> Dict:
        """
        Create the canonical entities.json registry.

        Args:
            employee_entities: List of employee entity dicts
            project_entities: List of project entity dicts
            product_entities: List of product entity dicts
            policy_entities: List of policy entity dicts

        Returns:
            Complete entity registry dict
        """
        print("\n[*] Building canonical entity registry...")

        # Add regulations from config
        regulation_entities = []
        for i, reg_config in enumerate(self.config['regulations'], 1):
            regulation_entities.append({
                'id': f"reg_{i:03d}",
                'name': reg_config['name'],
                'type': reg_config['type'],
                'full_name': reg_config['full_name']
            })

        entities = {
            'employees': employee_entities,
            'projects': project_entities,
            'products': product_entities,
            'policies': policy_entities,
            'regulations': regulation_entities
        }

        total_entities = sum(len(v) for v in entities.values())
        print(f"  ✓ Total entities: {total_entities}")
        print(f"    - Employees: {len(employee_entities)}")
        print(f"    - Projects: {len(project_entities)}")
        print(f"    - Products: {len(product_entities)}")
        print(f"    - Policies: {len(policy_entities)}")
        print(f"    - Regulations: {len(regulation_entities)}")

        return entities

    def validate_data(self,
                      employees_df: pd.DataFrame,
                      projects_df: pd.DataFrame,
                      products_df: pd.DataFrame,
                      policies_df: pd.DataFrame,
                      assignments_df: pd.DataFrame,
                      entities: Dict) -> bool:
        """
        Validate all generated data for consistency and integrity.

        Returns:
            True if all validations pass
        """
        print("\n[VALIDATION] Running integrity checks...")

        errors = []

        # Check for duplicate IDs
        all_ids = []
        for entity_type in entities.values():
            all_ids.extend([e['id'] for e in entity_type])

        if len(all_ids) != len(set(all_ids)):
            errors.append("❌ Duplicate entity IDs found")
        else:
            print("  ✓ No duplicate entity IDs")

        # Check employee manager references
        emp_ids = set(employees_df['employee_id'])
        invalid_managers = employees_df[
            employees_df['manager_id'].notna() &
            ~employees_df['manager_id'].isin(emp_ids)
            ]
        if len(invalid_managers) > 0:
            errors.append(f"❌ {len(invalid_managers)} invalid manager references")
        else:
            print("  ✓ All manager references valid")

        # Check policy owner references
        invalid_owners = policies_df[~policies_df['owner_id'].isin(emp_ids)]
        if len(invalid_owners) > 0:
            errors.append(f"❌ {len(invalid_owners)} invalid policy owner references")
        else:
            print("  ✓ All policy owner references valid")

        # Check assignment references
        proj_ids = set(projects_df['project_id'])
        invalid_proj_asgn = assignments_df[~assignments_df['project_id'].isin(proj_ids)]
        invalid_emp_asgn = assignments_df[~assignments_df['employee_id'].isin(emp_ids)]

        if len(invalid_proj_asgn) > 0:
            errors.append(f"❌ {len(invalid_proj_asgn)} invalid project references in assignments")
        if len(invalid_emp_asgn) > 0:
            errors.append(f"❌ {len(invalid_emp_asgn)} invalid employee references in assignments")

        if len(invalid_proj_asgn) == 0 and len(invalid_emp_asgn) == 0:
            print("  ✓ All assignment references valid")

        # Check for duplicate first names
        first_names = [e['first_name'] for e in entities['employees']]
        if len(first_names) != len(set(first_names)):
            errors.append("❌ Duplicate first names found (violates design requirement)")
        else:
            print("  ✓ All first names unique")

        # Check entities.json matches CSV counts
        if len(entities['employees']) != len(employees_df):
            errors.append("❌ Employee count mismatch between CSV and entities.json")
        if len(entities['projects']) != len(projects_df):
            errors.append("❌ Project count mismatch between CSV and entities.json")
        if len(entities['products']) != len(products_df):
            errors.append("❌ Product count mismatch between CSV and entities.json")
        if len(entities['policies']) != len(policies_df):
            errors.append("❌ Policy count mismatch between CSV and entities.json")

        if not errors:
            print("\n✅ All validation checks passed!")
            return True
        else:
            print("\n❌ Validation failed:")
            for error in errors:
                print(f"  {error}")
            return False

    def save_data(self,
                  employees_df: pd.DataFrame,
                  projects_df: pd.DataFrame,
                  products_df: pd.DataFrame,
                  policies_df: pd.DataFrame,
                  assignments_df: pd.DataFrame,
                  entities: Dict):
        """Save all data to CSV and JSON files."""
        print("\n[SAVING] Writing files to disk...")

        structured_dir = self.config['paths']['structured_dir']
        output_dir = self.config['paths']['output_dir']

        # Save CSVs
        employees_df.to_csv(f"{structured_dir}/employees.csv", index=False)
        print(f"  ✓ Saved employees.csv ({len(employees_df)} rows)")

        projects_df.to_csv(f"{structured_dir}/projects.csv", index=False)
        print(f"  ✓ Saved projects.csv ({len(projects_df)} rows)")

        products_df.to_csv(f"{structured_dir}/products.csv", index=False)
        print(f"  ✓ Saved products.csv ({len(products_df)} rows)")

        policies_df.to_csv(f"{structured_dir}/policies.csv", index=False)
        print(f"  ✓ Saved policies.csv ({len(policies_df)} rows)")

        assignments_df.to_csv(f"{structured_dir}/project_assignments.csv", index=False)
        print(f"  ✓ Saved project_assignments.csv ({len(assignments_df)} rows)")

        # Save entities.json
        entities_path = f"{output_dir}/entities.json"
        with open(entities_path, 'w') as f:
            json.dump(entities, f, indent=2)
        print(f"  ✓ Saved entities.json")

        print("\n✅ Phase 1 complete!")

    def run(self):
        """Execute complete structured data generation pipeline."""
        print("=" * 60)
        print("CodeFlow Knowledge Graph - Phase 1: Structured Data Generation")
        print("=" * 60)

        # Generate all data
        employees_df, employee_entities = self.generate_employees()
        projects_df, project_entities = self.generate_projects(employees_df)
        products_df, product_entities = self.generate_products()
        policies_df, policy_entities = self.generate_policies(employees_df)
        assignments_df = self.generate_project_assignments(employees_df, projects_df)

        # Build entity registry
        entities = self.generate_entities_json(
            employee_entities,
            project_entities,
            product_entities,
            policy_entities
        )

        # Validate
        if self.validate_data(
                employees_df,
                projects_df,
                products_df,
                policies_df,
                assignments_df,
                entities
        ):
            # Save
            self.save_data(
                employees_df,
                projects_df,
                products_df,
                policies_df,
                assignments_df,
                entities
            )
        else:
            print("\n❌ Data validation failed. Not saving files.")
            return False

        return True


def main():
    """Main entry point."""
    generator = StructuredDataGenerator("config.yaml")
    success = generator.run()

    if success:
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("  1. Review generated CSV files in data/structured/")
        print("  2. Examine entities.json for canonical entity registry")
        print("  3. Proceed to Phase 2: generate_reports.py")
        print("=" * 60)


if __name__ == "__main__":
    main()