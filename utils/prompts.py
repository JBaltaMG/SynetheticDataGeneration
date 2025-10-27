# prompts.py

# --- PROCUREMENT ---
PROCUREMENT_PROMPT = """
You are a procurement and industry expert. 
Generate a realistic ranked list of the top {count} procurement items, materials, and consumables 
commonly purchased by a Denmark-only, large-scale company like {company_name}, based on its industry and typical operations.

Output format:
- CSV with columns: name;proportionality;unit_price
- `name` = purchased item/material (avoid vague terms like "miscellaneous")
- `proportionality` = share of total procurement budget
- `unit_price` = cost per unit (DKK, number only)

Coverage requirements (all must appear at least once):
- Raw materials & base components
- Operational & maintenance supplies
- General equipment & consumables
- Office/admin products

Ranking rules:
- Sort by proportionality (desc)
- Top = most expensive raw/specialized inputs
- Middle = tools/spares/ops
- Bottom = office/admin

{contx}
{constraints}
"""

INVENTORY_PROMPT = f"""
You are a financial analyst and ERP data modeler.
Generate a realistic ranked list of {over_request_count} **Inventory items** for {company_name}, Denmark-only.

Fields:
- name = specific inventory line item
- inventory_segment = category (e.g., Raw Materials, Work in Progress, Finished Goods, Spare Parts, Consumables)
- proportionality = share of total inventory value (0–1 range)

Rules:
- Must map logically to balance sheet accounts of type "Inventory".
- Include both tangible materials (e.g., components, raw materials, WIP, finished goods) 
  and supporting stock (e.g., spare parts, packaging, maintenance inventory).
- Reflect the company’s likely industry (use realistic context from the company name).
- Rank by proportionality (descending).
- Avoid vague placeholders like "Miscellaneous Inventory" or "Other items".

{constraints}
""".strip()


# --- EBIT ITEMS ---
EBIT_PROMPT = f"""
You are a financial analyst and ERP data modeler.
Generate a realistic ranked list of {over_request_count} **EBIT items (Operating Income & Expenses)** for {company_name}, Denmark-only.

Fields:
- name = EBIT line item
- ebit_segment = category (e.g., Selling & Distribution, Administration, Production Overhead, Other Operating Income)
- proportionality = share of total EBIT impact (absolute value, 0–1 range)

Rules:
- Include both **operating costs** and **operating income** directly below gross profit, 
  excluding financing, tax, and extraordinary items.
- Costs include personnel, marketing, logistics, rent, IT, depreciation, etc.
- Income may include rental income, rebates, or gains on disposals.
- Must logically map to P&L accounts within the EBIT section of a CoA.
- Rank by proportionality (descending order).
- Avoid vague placeholders like “Other operating costs”.
{constraints}
""".strip()


# --- PRODUCTS ---
PRODUCTS_PROMPT = """
You are a product marketing and industry expert. 
Generate a realistic ranked list of the top {count} product categories 
that a Denmark-only, large-scale company like {company_name} would sell, 
based on its industry, brand identity, and market focus.

Output format:
- CSV with columns: name;proportionality;unit_price
- `name` = product/SKU category (e.g. "Running Shoes", not "Product A")
- `proportionality` = share of revenue
- `unit_price` = cost per unit (DKK, number only)

Composition rules:
- Mix of flagship, mid-range, and accessories
- Avoid vague placeholders
- Rank by proportionality (desc)

{contx}
{constraints}
"""

# --- ROLES ---
ROLES_PROMPT = """
You are an HR and industry expert. Generate {count} employee roles for a Denmark-only, large company like {company_name}.

Output format:
- CSV with one column: role_name
- Duplicates allowed (common roles)
- Rank by highest paid first

Distribution rules:
- 60–80% base titles (Software Engineer, Accountant, Sales Rep…)
- ≤10% with Senior/Lead/etc.
- ≤3 total C-level/VP
- Avoid hyper-granular or student roles

{contx}
{constraints}
"""

# --- SERVICES ---
SERVICES_PROMPT = """
You are a finance and procurement expert. 
Generate a realistic ranked list of the top {count} services, licenses, and fees 
commonly incurred by a Denmark-only, large-scale company like {company_name}.

Output format:
- CSV with: name;proportionality;unit_price
- `name` = specific service (e.g. "IT Consulting", "Microsoft 365 License")
- Avoid vague terms ("Miscellaneous")
- Include recurring/annual items

Composition rules:
- Top = expensive projects/outsourcing
- Middle = recurring departmental services
- Bottom = low-cost licenses/admin fees
- Balance across IT, legal, HR, facilities, etc.

{contx}
{constraints}
"""

# --- ACCOUNTS ---
ACCOUNTS_PROMPT = """
You are a financial accountant and ERP systems expert. 
Generate a realistic Chart of Accounts for {company_name} (Denmark-only, large company).

Each row = **leaf (posting) account** at level4
Columns:
- level3;level4;name;account_type

Rules:
- account_type = Revenue, Product Expense, Service Expense, Asset, Equity
- Many COGS accounts (for both products & services)
- Ignore payroll
- Balanced P&L + Balance Sheet
- Avoid vague "Miscellaneous"

{contx}
{constraints}
"""

# --- DEPARTMENTS ---
DEPARTMENTS_PROMPT = """
You are an HR and workforce distribution expert.
Generate {count} realistic departments for a Denmark-only, large company like {company_name}, each with a payroll share.

Output format:
- name;proportionality
- proportionality = decimal share of payroll (dot notation, e.g. 0.125)
- Sorted desc

Composition rules:
- General dept names (e.g. Production, Sales, Finance, IT, HR, Marketing…)
- Balanced mix across functions
- Avoid hyper-granular or temporal names

Distribution rules:
- Sum ≈ 1.0 (±0.01)
- Several mid-sized depts, none dominating unrealistically

{contx}
{constraints}
"""

# --- CUSTOMERS ---
CUSTOMERS_PROMPT = """
You are a B2B sales/marketing expert. 
Generate {count} realistic customers for {company_name}.

Fields:
- name
- customer_segment: Enterprise, SME, Government, Non-profit, Retail, Wholesale, Startup
- proportionality = share of customer base

Reflect realistic mix for Denmark-only company.

{contx}
{constraints}
"""

# --- VENDORS ---
VENDORS_PROMPT = """
You are a B2B procurement and supply chain expert. 
Generate {count} realistic vendors for {company_name}.

Fields:
- name
- vendor_type: Raw Materials, Equipment, IT Services, Logistics, Facilities, Office Supplies, Contract Labor, Consulting
- proportionality = vendor concentration

Bias toward critical categories, concentration implied by context.

{contx}
{constraints}
"""

# --- FINANCIAL ESTIMATION ---
FINANCIALS_PROMPT = """
Estimate the annual total revenue + operating costs (DKK) for {company_name}, Denmark-only.
Return a single integer (no text, no separators). Prefer explicit figures from context; otherwise estimate realistically.

{contx}
"""

# --- MEAN PAY ---
MEAN_PAY_PROMPT = """
What is the mean monthly pay in DKK for a {company_name} employee (incl. pension, vacation, benefits)?
Return one integer (no text, no separators). Prefer explicit numbers from context; otherwise estimate realistically.

{contx}
"""

# ------------------------
# Library of all prompts
# ------------------------
PROMPT_LIBRARY = {
    "procurement": PROCUREMENT_PROMPT,
    "products": PRODUCTS_PROMPT,
    "roles": ROLES_PROMPT,
    "services": SERVICES_PROMPT,
    "accounts": ACCOUNTS_PROMPT,
    "departments": DEPARTMENTS_PROMPT,
    "customers": CUSTOMERS_PROMPT,
    "vendors": VENDORS_PROMPT,
    "financials": FINANCIALS_PROMPT,
    "mean_pay": MEAN_PAY_PROMPT,
}
