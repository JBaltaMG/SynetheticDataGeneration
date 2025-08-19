import pandas as pd 
import utils.prompt_utils as prompt_utils
import utils.utils as utils
import generators.random_generators as random_generators
import numpy as np


def generate_procurement_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()

    over_request_count = int(np.floor(int(count) * 1.4))

    header = "name;proportionality;unit_price"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
    You are a procurement and industry expert. 
    Generate a realistic ranked list of the top {over_request_count} procurement items, materials, and consumables 
    commonly purchased by a Denmark-only, large-scale company like {company_name}, based on its industry and typical operations.

    Output format:
    - CSV with two columns: name;proportionality
    - `name` = specific purchased item/material (avoid vague terms like "miscellaneous" or "other")
    - `proportionality` = share of total procurement budget
    - `unit_price` = cost per unit (e.g., "1000"). The currency should be DKK. but only output the number 

    Coverage requirements (all must appear at least once):
    - Raw materials & base components
    - Operational & maintenance supplies
    - General equipment & consumables
    - Office/admin products

    Ranking rules:
    - Sort by proportionality in descending order
    - Top ranks: most expensive raw/specialized inputs
    - Middle ranks: tools, spares, operational items
    - Bottom ranks: low-cost office/admin supplies

    For context, here is a short version of the latest year-end report for {company_name}: 
    {ctxb}

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

    over_request_count = int(np.floor(int(count) * 1.4))

    header = "name;proportionality;unit_price"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
    You are a product marketing and industry expert. 
    Generate a realistic ranked list of the top {over_request_count} product categories 
    that a Denmark-only, large-scale company like {company_name} would sell, 
    based on its industry, brand identity, and market focus.


    Each row after the header:
    - `name` = realistic product/SKU category (broad but specific enough for revenue analysis; e.g., "Running Shoes", not "Product A")
    - `proportionality` = share of total sales revenue
    - `unit_price` = cost per unit (e.g., "1000"). The currency should be DKK. but only output the number 

    Composition rules:
    - Include a mix of high-revenue flagship lines, mid-range products, and low-cost accessories/services.
    - Avoid overly granular SKUs or vague placeholders like "Miscellaneous".
    - Rank the list in descending order of proportionality.

    For context, here is a short version of the latest year-end report for {company_name}:
    {ctxb}
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

def generate_roles_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.9):
    client = prompt_utils.get_openai_client()

    over_request_count = int(np.floor(int(count) * 1.4))

    header = "role_name"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    
    prompt = f"""
    You are an HR and industry expert. Generate {over_request_count} employee roles for a Denmark-only, large company like {company_name}.

    Output format:
    - CSV with a single column: role_name
    - Duplicates are ALLOWED and ENCOURAGED for common roles.
    - Rank by highest paid roles first. Any duplicates should be together. 

    Distribution requirements:
    - 60–80% must be generic base titles without seniority modifiers, e.g.:
    Software Engineer, Data Analyst, Consultant, Accountant, Sales Representative,
    Customer Support Specialist, Marketing Specialist, HR Generalist, Operations Coordinator,
    Procurement Specialist, Warehouse Associate, Project Manager, Business Analyst,
    QA Engineer, IT Support Specialist.
    - ≤10% may include seniority prefixes (Senior, Lead, Director, Head, Chief). Prefer none.
    - ≤3 entries total may be C‑level (CEO/CFO/CTO/etc.) or VP.
    - Avoid hyper‑granular one-offs; repetition of core roles is preferred.

    Style constraints:
    - Avoid prefixes: Senior, Lead, Director, Head, Chief, Principal, Staff — unless within the ≤10% cap.
    - Avoid internship/student titles.
    - Keep roles realistic for Denmark-only operations; no global country managers.

    For context, here is a short version of the lastest year-end report for {company_name}:
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

    over_request_count = int(np.floor(int(count) * 1.4))

    header = "name;proportionality;unit_price"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
    You are a finance and procurement expert. 
    Generate a realistic ranked list of the top {over_request_count} services, licenses, and fees 
    commonly incurred by a Denmark-only, large-scale company like {company_name}.

    Each row after the header:
    - `name` = specific service, license, or fee (e.g., "IT Consulting", "Microsoft 365 License", "Legal Retainer")
    - Avoid vague terms like "Miscellaneous" or "Various".
    - Include annual/temporal entries such as "Annual IT Audit".
    - `proportionality` = share of total sales revenue
    - `unit_price` = cost per unit (e.g., "1000"). The currency should be DKK. but only output the number.

    Composition rules:
    - Top ranks: expensive projects, enterprise retainers, major outsourcing contracts.
    - Middle ranks: recurring professional services and departmental support.
    - Bottom ranks: standardized low-cost licenses and administrative fees.
    - Include a balance across IT, legal, marketing, HR, facilities, and general admin.

    For context, here is a short version of the latest year-end report for {company_name}:
    {ctxb}

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

def generate_accounts_llm(company_name: str, count: int = 30, model: str = "gpt-4.1", temp: float=0.8) -> pd.DataFrame:
    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))

    header = "level3;level4;name;account_type"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
    You are a financial accountant and ERP systems expert. Generate a realistic Chart of Accounts for {company_name} (Denmark-only, large company).

    Each row describes a **leaf (posting) account** at level4:
    l1_code: General posting codes, i.e. 1000, 2000, 3000, etc 
    account_type: "Revenue","Product Expense","Service Expense","Asset","Equity". THEY NEED TO MATCH THE BUSINESS CONTEXT.
    l2_code: More specific posting codes, i.e. 1100, 2100, etc. 
    l2_name: category (e.g., "Cash & Cash Equivalents", "Product COGS", "Service Delivery Costs", "Marketing")
    account_id = the posting account id
    name = the posting account name (concise, no codes)
    WE NEED MANY COGS. For both products, services.     
    Rules:
    - Ignore payroll accounts.
    - Clearly separate **Product Expense** vs **Service Expense** under Expenses.
    - Include a balanced P&L and Balance Sheet spread (not only P&L).
    - Keep names realistic; avoid vague buckets like "Miscellaneous".

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

def generate_departments_llm(company_name: str, count: int = 10, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()
    header = "name;proportionality"
    over_request_count = int(np.floor(int(count) * 1.4))

    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
    You are an HR and workforce distribution expert.
    Generate {over_request_count} realistic departments for a Denmark-only, large company like {company_name}, each with a payroll share.

    STRICT OUTPUT:
    - Fields (header must match exactly): name;proportionality
    - proportionality = decimal share of total payroll using a dot (e.g., 0.125)
    - Sort rows by proportionality in descending order.

    Composition rules:
    - Prefer **general department names** (avoid seniority or role-specific titles).
    - Include a balanced mix across core functions. Strong candidates:
    "Operations","Production","Manufacturing","R&D","Engineering","IT","Data & Analytics",
    "Sales","Marketing","Customer Service","Finance","Procurement","Logistics",
    "HR","Legal & Compliance","Facilities","Quality Assurance","Strategy/PMO".
    - Avoid hyper-granular teams (e.g., "Backend Platform Team") and avoid temporal qualifiers.

    Distribution rules:
    - Shares should be realistic for a large Danish company.
    - Sum of proportionality should be approximately 1.0 (±0.01).
    - Allow multiple mid-sized departments; do not make one department dominate unrealistically.

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

def generate_customers_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))
    header = "name;customer_segment;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)

    prompt = f"""
    You are a B2B sales/marketing expert. Generate {over_request_count} realistic customers for {company_name}.
    Fields:
    - name
    - customer_segment: Enterprise, SME, Government, Non-profit, Retail, Wholesale, Startup
    - proportionality: the proportionality of this customer 
    Reflect size/segment mix implied by the context.


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

def generate_vendors_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))
    header = "name;vendor_type;proportionality"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    prompt = f"""
    You are a B2B procurement and supply chain expert. Generate {over_request_count} realistic vendors for {company_name}.
    Fields:
    - name
    - vendor_type: Raw Materials, Equipment, IT Services, Logistics, Facilities, Office Supplies, Contract Labor, Consulting
    - proportionality: the proportionality of this vendor.
    Bias critical categories and concentration based on the context.

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

For context, here is a short version of the lastest year-end report for {company_name}: 
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

    For context, here is a short version of the lastest year-end report for {company_name}: 
    {ctxb}
    """.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful payroll analyst."},
                  {"role": "user", "content": prompt}],
        temperature=temp,
    )
    return int(response.choices[0].message.content)
