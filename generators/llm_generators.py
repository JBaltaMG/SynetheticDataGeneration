import pandas as pd 
import utils.prompt_utils as prompt_utils
import utils.utils as utils
import generators.random_generators as random_generators
import numpy as np

def generate_bus_llm(company_name: str, count: int = 100, model: str = "gpt-4.1", temp: float = 0.8):
    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))
    header = "bu_id;bu_name;bu_type;country;company_code"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    PROMPT_BUSINESS_UNITS = f"""
    You are creating a realistic internal org structure for a company.

    Company: {company_name}

    Task:
    Generate 10-15 business units and departments that reflect how this company would actually operate (production sites, regional sales orgs, HQ functions, logistics hubs, shared services, etc.).

    Return ONLY a semicolon-separated CSV with the following columns in this exact order:
    {header}

    Definitions:
    - BU_ID: stable ID like BU001, BU002, ...
    - BU_Name: human label, e.g. "{company_name} Retail", "{company_name} Factory". ALSO WE need a headquarter in the correct country. JUST CALLED "{company_name} HQ".
    - BU_Type: one of [Factory, Retail, HQ, Licensing, Shared Service, Online, Distribution]
    - Country: realistic country/region for that BU. Max of 5 countries total. 
    - CompanyCode: For each country make a company Code. Start at 1000 and increment by 250 for each new country. 

    Rules:
    - Make sure there is at least one HQ / corporate finance unit.
    - Make sure there are both commercial (sales/retail) and production/supply-side units.
    - IDs must be unique.
    {constraints}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful data assistant and B2B vendor segmentation expert."},
                  {"role": "user", "content": PROMPT_BUSINESS_UNITS}],
        temperature=temp,
    )
    
    return prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)

def generate_line_items_llm(company_name: str, 
                            count: int = 100, 
                            category_name: str = "COGS", 
                            financial_total: float = 100000.0, 
                            df_business_units: pd.DataFrame = pd.DataFrame(),
                            df_parties: pd.DataFrame = pd.DataFrame(),
                            df_accounts: pd.DataFrame = pd.DataFrame(),
                            model: str = "gpt-4.1", 
                            temp: float = 0.5):
    
    """
    Generate COGS (Cost of Goods Sold) items for a company.
    Each item should map to Product Expense or Service Expense accounts in the COA.
    """

    accounts_subset_csv = df_accounts.iloc[:, :2].to_csv(index=False, header=False, sep=";")
    business_units_csv = df_business_units.iloc[:, :2].to_csv(index=False, header=False, sep=";")
    parties_csv = df_parties.iloc[:, :3].to_csv(index=False, header=False, sep=";")

    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))

    header = "document_number;posting_date;company;bu_id;bu_name;party_id;party_name;account_id;account_name;item_name;proportionality;unit_price;markdown;category"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    PROMPT_LINES = f"""
    You are generating aggregated GL driver lines for synthetic financial data.
    This is NOT journal entries yet. This is the template that will later be exploded
    into many detailed postings with dates, document numbers, etc.

    Company: {company_name}
    Category to generate: {category_name}   # e.g. Revenue, COGS, FixedCost, EBIT
    Number of rows to generate (before later down-splitting): {over_request_count}

    ACCOUNTS (only use these AccountKeys for this category):
    {accounts_subset_csv}

    BUSINESS UNITS (use these bu_id values only):
    {business_units_csv}

    PARTIES (customers, vendors, internal units):
    {parties_csv}

    YOUR TASK
    Generate {over_request_count} high-level economic driver lines for the given category {category_name}.

    For each item, there must be 2-3 lines. Proportionality should be split accordingly.
    ONLY 5% of all lines must be intercompany. 
    
    COLUMNS AND ORDER
    You MUST output a semicolon-separated CSV with columns in this exact order:

    {header}
    
    Column definitions:
    - bu_id:
    - Must match one of the bu_id values from BUSINESS UNITS.
    - Pick whichever BU is most natural for that driver (e.g. retail cost -> retail BU).

    - party_id:
    - Revenue:
        - If AccountKey is an intercompany revenue account, party_id must be an INTERNAL_BU from PARTIES. IMPORTANT: MAX {over_request_count*0.05} or 5% internal sales total. 
        - Otherwise use a CUSTOMER from PARTIES.
    - COGS:
        - If AccountKey is an intercompany COGS account, party_id must be an INTERNAL_BU. IMPORTANT: MAX {over_request_count*0.05} or 5% internal sales total. 
        - Otherwise use a VENDOR from PARTIES.
    - FixedCost:
        - Can be blank unless there's a clear vendor/counterparty (e.g. "External legal services").
    - EBIT (other income/expense categories):
        - Can be blank unless it's obviously a financing/royalty counterparty.

    - AccountKey:
    - Must be copied from the provided ACCOUNTS list.
    - Only use AccountKeys valid for this category:
        - Revenue  → 4001–4009
        - COGS     → 4003, 4006, 4009
        - FixedCost→ 5001–5027
        - EBIT     → 6001–6503
        - BalanceSheet → 1001–3005

    - AccountName:
    - Must be copied from 'name' in the accounts list for that AccountKey.

    - item_name:
    - BE specific. Must look like a real item or service description. ex. "Consulting services", "Plastic gloves" etc.
    - MUST NOT include dates, months, regions, shipment references, batch IDs, PO numbers, 'January', 'Copenhagen', etc.
    - MUST NOT include "for" phrases or detailed invoice descriptions.

    `proportionality` rules:
    - `proportionality` = share of total budget
    -  Represents how large this driver is relative to the TOTAL for this category.
    -  Must be larger than 1
    
    - unit_price (sales):
    - unit_price of the item in DKK. (Unit_price must be larger than 10Dkk and lower than 100000 Dkk) 

    - markdown: 
    - Conversion of unit_price sales to cost.
    - E.g. 0.25 = 25% markdown on cost.

    - category:
    - Must equal {category_name} exactly, for every row.

    FINAL OUTPUT RULES
    - Output ONLY CSV rows, one row per driver line.
    - Use semicolons as separators.
    - Do NOT include headers.
    - Do NOT include document_number.
    - Do NOT include amount_DKK.
    - Do NOT include unit_price.
    - Do NOT include explanations, notes, or markdown fences.

    {constraints}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst and ERP mapping assistant."},
            {"role": "user", "content": PROMPT_LINES},
        ],
        temperature=temp,
    )

    df_lines = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_lines = utils.convert_column_to_percentage(df_lines, "proportionality", scale=1.0)
    df_lines["annual_spend"] = np.round(df_lines["proportionality"] * financial_total, -2)
    df_lines = df_lines.drop(columns=["account_name", "document_number", "posting_date", "proportionality", "company"])
    return df_lines

def tailor_coa_names_llm(
    df_coa: pd.DataFrame,
    company_name: str,
    model: str = "gpt-4.1",
    temp: float = 0.7,
) -> pd.DataFrame:
    """Customize Chart of Accounts names to reflect a specific company using an LLM."""

    if "AccountKey" not in df_coa.columns or "name" not in df_coa.columns:
        raise ValueError("DataFrame must contain 'AccountKey' and 'name' columns.")

    df_coa = df_coa.copy()
    account_key_strings = df_coa["AccountKey"].astype(str)

    header = "AccountKey;name"
    constraints = prompt_utils.get_standard_constraints(header, len(df_coa))
    ctxb = prompt_utils._ctx_block(company_name)

    account_rows = "\n".join(
        f"{key};{name}"
        for key, name in zip(account_key_strings, df_coa["name"].astype(str))
    )

    prompt = f"""
    {ctxb}

    You are an ERP accountant updating the Chart of Accounts language for {company_name}.
    You will receive account keys with their current names. Rewrite ONLY the names so they
    sound authentic for {company_name}'s industry, brands, and terminology.

    Rules:
    - Keep every AccountKey exactly the same and preserve the order.
    - Produce concise, professional account names with no codes or numbering.
    - Do not add or remove accounts. Return exactly one row for each input.
    - Ensure the vocabulary reflects {company_name}'s products, services, and operations.

    Input accounts (AccountKey;name):
    {account_rows}

    Respond with a semicolon-delimited CSV using the exact header:
    {header}

    {constraints}
    """.strip()

    client = prompt_utils.get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a meticulous ERP and accounting assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
    )

    df_names = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, len(df_coa))
    if isinstance(df_names, int) and df_names == -1:
        raise ValueError("LLM response did not contain the expected number of rows.")
    if df_names.empty:
        raise ValueError("LLM response could not be parsed into account names.")

    df_names["AccountKey"] = df_names["AccountKey"].astype(str)
    updated_names = df_names.set_index("AccountKey")["name"]

    new_names = account_key_strings.map(updated_names)
    df_coa["name"] = new_names.fillna(df_coa["name"])

    return df_coa


def generate_parties(company_name: str, 
                            count: int = 100, 
                            df_business_units: pd.DataFrame = pd.DataFrame(),
                            model: str = "gpt-5", 
                            temp: float = 1):
    
    """
    Generate COGS (Cost of Goods Sold) items for a company.
    Each item should map to Product Expense or Service Expense accounts in the COA.
    """
    business_units_csv = df_business_units.iloc[:, :4].to_csv(index=False, header=False, sep=";")
    
    client = prompt_utils.get_openai_client()
    over_request_count = int(np.floor(int(count) * 1.4))
    
    header = "party_ID;party_name;party_type;party_country"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    PROMPT_PARTIES = f"""
    You are creating master data for all counterparties in this company.

    Company: {company_name}

    Internal business units (BU master data):
    {business_units_csv}

    Task:
    1. For each internal BU, create a row where that BU is treated as a party.
    2. Also create external customers (distributors, retailers, channels).
    3. Also create external vendors (materials suppliers, logistics, energy, maintenance, IT services).

    Generate {over_request_count} rows TOTAL across all types. About 47.5% for custumers, 47.5% for vendors, and 5% for internal sales.

    Return ONLY a semicolon-separated CSV with columns in this exact order:
    {header}

    Where:
    - party_ID:
    - INTERNAL_BU => bu_id for the party. Can be found in the provided CSV. - MAX {over_request_count*0.05} or 5% internal sales total. 
    - CUSTOMER    => "CUS###"
    - VENDOR      => "VEN###"
    - party_Type is exactly one of [INTERNAL_BU, CUSTOMER, VENDOR]
    - INTERNAL_BU rows must include ALL internal business units given above.
    - IF INTERNAL_BU: It cannot be identical to bu_id on the same line. They must sell between units. 
    - party_name for INTERNAL_BU must match BU_Name exactly.
    - party_country: country of origin of the party. 
    - No commentary, no markdown, only CSV.

    {constraints}

    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst and ERP mapping assistant."},
            {"role": "user", "content": PROMPT_PARTIES},
        ],
        temperature=temp,
    )

    df_parties = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)

    return df_parties

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


def generate_roles_and_departments_llm(company_name: str, 
                       count: int = 100, 
                       model: str = "gpt-4.1", 
                       df_business_units: pd.DataFrame = pd.DataFrame(),
                       temp: float = 0.9):
    client = prompt_utils.get_openai_client()

    over_request_count = int(np.floor(int(count) * 1.4))

    header = "role_name;department;department_id;bu_id"
    constraints = prompt_utils.get_standard_constraints(header, over_request_count)
    ctxb = prompt_utils._ctx_block(company_name)

    business_units_csv = df_business_units.iloc[:, :2].to_csv(index=False, header=False, sep=";")
    
    prompt = f"""
    You are an HR and industry expert. Generate {over_request_count} employee roles and departments for a large company like {company_name}.

    Internal business units (BU master data):
    {business_units_csv}
    
    
    Output format:
    - CSV with columns: {header}
    - Duplicates are ALLOWED and ENCOURAGED for common roles.
    - Rank by highest paid roles first. Any duplicates should be together. 
    
    COLS: 
    - role_name: realistic job title. Must include CEO and CFO.
    - department: name of the department the role belongs to. Ex. Data Scientist -> R&D. Must be an actual realistic department name.
    - department_id: stable ID for the department, e.g. "DPT001", "DPT002".
    - bu_id: assign each role to one of the bu_id's from the internal business units list. Ex. CFO -> HQ BU.

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
    df_roles = prompt_utils.parse_and_truncate_csv(response.choices[0].message.content, count)
    df_departments = df_roles.iloc[:,-3:].drop_duplicates("department").reset_index(drop=True)
    df_roles = df_roles.drop(columns=["department", "bu_id"])
    return df_roles, df_departments
