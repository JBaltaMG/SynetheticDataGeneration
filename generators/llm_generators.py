import pandas as pd 
import utils.prompt_utils as prompt_utils
import utils.utils as utils
import generators.random_generators as random_generators


def generate_procurement_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count) * 1.4
    header = "name;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are a procurement and industry expert. Generate a realistic ranked list of the top {over_request_count} procurement items, materials, and consumables 
commonly purchased by a company like {company_name}, based on its industry and typical operations.

Each row:
- name: specific purchased item/material
- proportionality: % of total procurement budget (0–100)

Coverage:
- Raw materials & base components
- Operational & maintenance supplies
- General equipment & consumables
- Office/admin products

Ranking:
- Top: most expensive raw/specialized inputs
- Middle: tools, spares, operational items
- Bottom: low-cost office/admin supplies

{ctxb}
Rank by proportionality in descending order.

{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data analyst and industry expert."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_procurement = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_procurement = utils.convert_column_to_percentage(df_procurement, "proportionality", scale=1.0)
    return df_procurement

def generate_sales_products_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count * 1.4)
    header = "name;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are a product marketing and industry expert. Generate a realistic ranked list of the top {over_request_count} products
that a company like {company_name} would sell, based on its industry, brand identity, and market focus.

Each row:
- name: realistic product/SKU category (categorical, not overly specific)
- proportionality: % of total sales revenue (0–100)

Ensure a mix of high-revenue flagships, mid-range products, and low-cost accessories/services.
{ctxb}
Rank the list by proportionality in descending order.

{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data analyst and industry expert."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_products = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_products = utils.convert_column_to_percentage(df_products, "proportionality", scale=1.0)
    return df_products

def generate_roles_llm(company_name: str, count: int = 100, model: str = "gpt-4o", temp: float = 0.1):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count) * 1.4
    header = "role_name"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are an HR and industry expert. Generate {over_request_count} realistic employee roles for a company like {company_name}.
Bias titles/functions indicated by the context (e.g., R&D intensity, retail footprint, digital focus).
Each row: role_name
Rank by highest monthly salary first.
{ctxb}
{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data analyst and HR expert."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    return prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)

def generate_services_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()

    over_request_count = int(count) * 1.4
    header = "name;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are a finance and procurement expert. Generate a realistic ranked list of the top {over_request_count} services, licenses, and fees 
commonly incurred by a company like {company_name}.

Each row:
- name: specific service/fee (e.g. 'IT Consulting', 'Microsoft 365 License', 'Legal Retainer')
- proportionality: % of total procurement budget (0–100)
Do not make annual entries.

Top: expensive projects/enterprise retainers
Middle: regular professional services and departmental support
Bottom: standardized low-cost licenses/admin services
{ctxb}
Rank by proportionality in descending order.

{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data analyst and finance expert."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_services = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_services = utils.convert_column_to_percentage(df_services, "proportionality", scale=1.0)
    return df_services

def generate_accounts_llm(company_name: str, count: int = 30, model: str = "gpt-4o", temp: float=0.5) -> pd.DataFrame:
    client = prompt_utils.get_openai_client()
    over_request_count = int(count) * 1.4
    header = "name;account_type"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are a financial accountant and ERP systems expert. Generate a realistic Chart of Accounts for {company_name}.
Return:
- name
- account_type: "Revenue","Product Expense","Service Expense","Asset","Equity"
Ignore payroll accounts.
Ensure diverse P&L and BS accounts; clearly split Product vs Service expenses.
{ctxb}
{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful ERP and accounting assistant."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_accounts = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_accounts["account_id"] = random_generators.generate_account_ids(df_accounts["account_type"])
    return df_accounts

def generate_departments_llm(company_name: str, count: int = 10, model: str = "gpt-4o", temp: float = 0.5):
    client = prompt_utils.get_openai_client()
    header = "name;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are an HR and workforce distribution expert. Generate {count} realistic departments for {company_name} with payroll share.
Fields:
- name
- proportionality: decimal share of total payroll (sums ~1.0)
Use context signals (e.g., manufacturing vs retail vs digital) to bias department mix.
{ctxb}
{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful real estate and HR data assistant."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_offices = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_offices.insert(0, "department_id", range(100, len(df_offices) + 100))
    df_offices = utils.convert_column_to_percentage(df_offices, "proportionality", scale=1.0)
    return df_offices

def generate_customers_llm(company_name: str, count: int = 100, model: str = "gpt-4o", temp: float = 0.3):
    client = prompt_utils.get_openai_client()
    header = "name;customer_segment;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are a B2B sales/marketing expert. Generate {count} realistic customers for {company_name}.
Fields:
- name
- customer_segment: Enterprise, SME, Government, Non-profit, Retail, Wholesale, Startup
- proportionality: decimal revenue share (sums ~1.0)
Reflect size/segment mix implied by the context.
{ctxb}
{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data assistant and B2B customer segmentation expert."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_customers = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_customers.insert(0, "customer_id", range(10, len(df_customers) + 10))
    df_customers = utils.convert_column_to_percentage(df_customers, "proportionality", scale=1.0)
    return df_customers

def generate_vendors_llm(company_name: str, count: int = 100, model: str = "gpt-4o", temp: float = 0.3):
    client = prompt_utils.get_openai_client()
    header = "name;vendor_type;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
You are a B2B procurement and supply chain expert. Generate {count} realistic vendors for {company_name}.
Fields:
- name
- vendor_type: Raw Materials, Equipment, IT Services, Logistics, Facilities, Office Supplies, Contract Labor, Consulting
- proportionality: decimal spend share (sums ~1.0)
Bias critical categories and concentration based on the context.
{ctxb}
{constraints}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data assistant and B2B vendor segmentation expert."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    df_vendors = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_vendors.insert(0, "vendor_id", range(20, len(df_vendors) + 20))
    df_vendors = utils.convert_column_to_percentage(df_vendors, "proportionality", scale=1.0)
    return df_vendors

def estimate_financials_llm(company_name: str, model: str = "gpt-4o", temp: float = 0):
    client = prompt_utils.get_openai_client()
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
Estimate the **annual total revenue + operating costs (DKK)** for {company_name}, Denmark-only.
Return a single integer (no text, no separators). Prefer explicit figures from context; otherwise estimate from scale/industry.
{ctxb}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful payroll analyst."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    total_finances = int(response.choices[0].message.content)

    payroll  = int(total_finances * 0.35)
    products = int(total_finances * 0.45)
    services = int(total_finances * 0.15)
    overhead = int(total_finances * 0.05)

    return {
        "total_finances": total_finances,
        "payroll": payroll,
        "products": products,
        "services": services,
        "overhead": overhead
    }

def estimate_mean_pay_llm(company_name: str, model: str = "gpt-4o", temp: float = 0):
    client = prompt_utils.get_openai_client()
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
What is the mean monthly pay in DKK for a {company_name} employee (incl. pension, vacation, benefits)?
Return just one integer (no text, no separators). Prefer explicit salary numbers from context; otherwise estimate realistically for Denmark/industry.
{ctxb}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful payroll analyst."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    return int(response.choices[0].message.content)
